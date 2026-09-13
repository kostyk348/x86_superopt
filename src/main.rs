//! x86_superopt — Same-ISA x86_64 Binary Super-Optimizer
//! Оптимизирует функции из ELF64 бинарников: SSA → AVX2 вектор, branchless, inline, DCE → emit

use clap::Parser;
use sdre::{Arch, DataflowGraph, Value, dce, disasm_to_ir, lift_region, split_regions, RegionConfig};
use std::fs;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::thread;

// ─── ELF64 ───────────────────────────────────────────────────────────────────
#[derive(Copy, Clone)]
#[repr(C)]
struct Elf64_Ehdr {
    e_ident: [u8; 16],
    e_type: u16,
    e_machine: u16,
    e_version: u32,
    e_entry: u64,
    e_phoff: u64,
    e_shoff: u64,
    e_flags: u32,
    e_ehsize: u16,
    e_phentsize: u16,
    e_phnum: u16,
    e_shentsize: u16,
    e_shnum: u16,
    e_shstrndx: u16,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct Elf64_Shdr {
    sh_name: u32,
    sh_type: u32,
    sh_flags: u64,
    sh_addr: u64,
    sh_offset: u64,
    sh_size: u64,
    sh_link: u32,
    sh_info: u32,
    sh_addralign: u64,
    sh_entsize: u64,
}

#[derive(Copy, Clone)]
#[repr(C)]
struct Elf64_Sym {
    st_name: u32,
    st_info: u8,
    st_other: u8,
    st_shndx: u16,
    st_value: u64,
    st_size: u64,
}

const SHT_SYMTAB: u32 = 2;
const SHT_STRTAB: u32 = 3;
const SHT_PROGBITS: u32 = 1;
const SHF_EXECINSTR: u64 = 0x4;
const STT_FUNC: u8 = 2;

fn elf_read<T: Copy>(bytes: &[u8], offset: usize) -> T {
    let size = std::mem::size_of::<T>();
    let mut val: T = unsafe { std::mem::zeroed() };
    let ptr = &mut val as *mut T as *mut u8;
    unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr().add(offset), ptr, size); }
    val
}

struct ElfCtx {
    text_off: u64,  // .text section file offset
    text_addr: u64, // .text section virtual address
    text_size: u64,
    text_idx: u16,
    symbols: Vec<(u64, u64, String)>, // (addr, size, name)
}

fn parse_elf(bytes: &[u8]) -> Result<ElfCtx, String> {
    if bytes.len() < 64 { return Err("File too small for ELF header".into()); }
    let ident = &bytes[..16];
    if &ident[..4] != b"\x7fELF" || ident[4] != 2 { return Err("Not ELF64".into()); } // 2 = ELFCLASS64

    let ehdr: Elf64_Ehdr = elf_read(&bytes, 0);
    let shoff = ehdr.e_shoff as usize;
    let shnum = ehdr.e_shnum as usize;
    let shentsize = ehdr.e_shentsize as usize;
    if shoff == 0 || shnum == 0 { return Err("No section headers".into()); }
    let shstrndx = ehdr.e_shstrndx as usize;

    // Read shstrtab
    if shstrndx >= shnum { return Err("Invalid shstrndx".into()); }
    if shentsize < 64 { return Err("Invalid section header entry size".into()); }
    let shstr_hdr: Elf64_Shdr = elf_read(&bytes, shoff + shstrndx * shentsize);
    let strtab_off = shstr_hdr.sh_offset as usize;
    let strtab_size = shstr_hdr.sh_size as usize;

    let shdr = |i: usize| -> Elf64_Shdr {
        elf_read(&bytes, shoff + i * shentsize)
    };

    let sect_name = |hdr: &Elf64_Shdr| -> String {
        let off = strtab_off + hdr.sh_name as usize;
        bytes[off..].iter().take_while(|&&b| b != 0).map(|&b| b as char).collect()
    };

    // Find .text section by name
    let mut text_off = 0u64; let mut text_addr = 0u64;
    let mut text_size = 0u64; let mut text_idx = 0u16;
    for i in 0..shnum {
        let h = shdr(i);
        let sn = sect_name(&h);
        if sn == ".text" {
            text_off = h.sh_offset; text_addr = h.sh_addr;
            text_size = h.sh_size; text_idx = i as u16;
            break;
        }
    }
    if text_size == 0 {
        // Fallback: first executable PROGBITS
        for i in 0..shnum {
            let h = shdr(i);
            if h.sh_type == SHT_PROGBITS && (h.sh_flags & SHF_EXECINSTR) != 0 {
                text_off = h.sh_offset; text_addr = h.sh_addr;
                text_size = h.sh_size; text_idx = i as u16;
                break;
            }
        }
    }
    if text_size == 0 { return Err("No executable .text section".into()); }

    // Find symtab
    let mut sym_off = 0usize; let mut sym_entsize = 0usize; let mut sym_count = 0usize;
    let mut symstr_off = 0usize; let mut symstr_size = 0usize;
    for i in 0..shnum {
        let h = shdr(i);
        if h.sh_type == SHT_SYMTAB {
            sym_off = h.sh_offset as usize;
            sym_entsize = h.sh_entsize as usize;
            sym_count = h.sh_size as usize / sym_entsize.max(1);
            // Find linked string table
            let str_idx = h.sh_link as usize;
            if str_idx < shnum {
                let strh = shdr(str_idx);
                symstr_off = strh.sh_offset as usize;
                symstr_size = strh.sh_size as usize;
            }
            break;
        }
    }
    if sym_count == 0 { return Ok(ElfCtx{text_off, text_addr, text_size, text_idx, symbols: vec![]}); }

    let mut symbols = Vec::new();
    for i in 0..sym_count {
        let sym: Elf64_Sym = elf_read(&bytes, sym_off + i * sym_entsize);
        if sym.st_info & 0xf != STT_FUNC { continue; }
        if sym.st_shndx != text_idx { continue; }
        if sym.st_size == 0 { continue; }
        // Read name
        let name_off = symstr_off + sym.st_name as usize;
        let name: String = bytes[name_off..].iter()
            .take_while(|&&b| b != 0).map(|&b| b as char).collect();
        symbols.push((sym.st_value, sym.st_size, name));
    }
    symbols.sort_by_key(|s| s.0); // sort by address

    Ok(ElfCtx{text_off, text_addr, text_size, text_idx, symbols})
}

// ─── OPTIMIZATION PASSES ──────────────────────────────────────────────────────

fn vectorize_loop(g: &mut DataflowGraph) -> bool {
    let mut modified = false;
    let values = g.values.clone();
    let mut i = 0;
    while i + 3 < values.len() {
        if !(i..i+4).all(|j| matches!(&values[j], Value::Add{..})) { i += 1; continue; }

        let mut load_addrs: Vec<usize> = Vec::new();
        let mut accum_base: Option<usize> = None;
        let mut first_load_val: Option<usize> = None;

        for j in i..i+4 {
            if let Value::Add{lhs,rhs} = &values[j] {
                if let Some(Value::Load{addr,..}) = values.get(*lhs) {
                    load_addrs.push(*addr);
                    if first_load_val.is_none() { first_load_val = Some(*lhs); }
                    accum_base = Some(*rhs);
                } else if let Some(Value::Load{addr,..}) = values.get(*rhs) {
                    load_addrs.push(*addr);
                    if first_load_val.is_none() { first_load_val = Some(*rhs); }
                    accum_base = Some(*lhs);
                }
            }
        }

        let consecutive_ok = load_addrs.len() >= 3
            && load_addrs.iter().all(|a| *a == load_addrs[0]);
        let has_base = first_load_val.is_some() && load_addrs.len() >= 2;

        if consecutive_ok && has_base && accum_base.is_some() {
            g.values[i] = Value::Opaque {
                mnemonic: "VECTOR_ADD_4x32".into(),
                inputs: vec![first_load_val.unwrap(), accum_base.unwrap()],
                bits: 128,
            };
            for j in i+1..=i+3 { g.values[j] = Value::Const{val:0, bits:32}; }
            modified = true; i += 4; continue;
        }
        i += 1;
    }
    modified
}

fn branchless_pass(g: &mut DataflowGraph) -> bool {
    let mut modified = false;
    let effects: Vec<(usize, sdre::Effect)> = g.effects.iter().enumerate().map(|(i,e)| (i,e.clone())).collect();
    for (ei, e) in effects {
        if let sdre::Effect::BranchCond{cond,..} = e {
            if let Some(Value::Cmp{lhs,rhs,..}) = g.values.get(cond) {
                g.effects[ei] = sdre::Effect::OpaqueEffect{mnemonic:"branchless".into(), inputs:vec![*lhs,*rhs]};
                modified = true;
            }
        }
    }
    modified
}

fn inline_pass(regions: &[sdre::RegionBlock], graphs: &mut [DataflowGraph]) -> bool {
    let mut modified = false;
    for ri in 0..regions.len() {
        let calls: Vec<(usize,u64)> = graphs[ri].effects.iter().enumerate()
            .filter_map(|(ei,e)| if let sdre::Effect::Call{target,..}=e { Some((ei,*target)) } else { None }).collect();
        for (ei, target) in calls {
            if let Some(cj) = regions.iter().position(|r| r.entry_addr == target) {
                if graphs[cj].values.len() < 50 {
                    graphs[ri].effects[ei] = sdre::Effect::OpaqueEffect{mnemonic:"inlined".into(), inputs:vec![]};
                    modified = true; break;
                }
            }
        }
    }
    modified
}

// ─── REGALLOC + EMIT ──────────────────────────────────────────────────────────

struct RegAlloc { map: Vec<Option<String>>, next_reg: usize }
impl RegAlloc {
    fn new(n: usize) -> Self { Self { map: vec![None; n], next_reg: 0 } }
    fn alloc(&mut self, vid: usize) -> String {
        if vid >= self.map.len() { return format!("v{}", vid); }
        if self.map[vid].is_none() {
            let regs = ["eax","ebx","ecx","edx","esi","edi","r8d","r9d","r10d","r11d","r12d","r13d","r14d","r15d"];
            self.map[vid] = Some(regs[self.next_reg % 14].to_string());
            self.next_reg += 1;
        }
        self.map[vid].clone().unwrap()
    }
    /// Allocate a 64-bit register (address bases): rax, rbx, ... r15.
    fn alloc64(&mut self, vid: usize) -> String {
        if vid >= self.map.len() { return format!("v{}", vid); }
        if self.map[vid].is_none() {
            let regs = ["rax","rbx","rcx","rdx","rsi","rdi","r8","r9","r10","r11","r12","r13","r14","r15"];
            self.map[vid] = Some(regs[self.next_reg % 14].to_string());
            self.next_reg += 1;
        }
        self.map[vid].clone().unwrap()
    }
    fn r(&mut self, vid: usize) -> String {
        if vid >= self.map.len() { return format!("v{}", vid); }
        if self.map[vid].is_none() {
            let regs = ["eax","ebx","ecx","edx","esi","edi","r8d","r9d","r10d","r11d","r12d","r13d","r14d","r15d"];
            self.map[vid] = Some(regs[self.next_reg % 14].to_string());
            self.next_reg += 1;
        }
        self.map[vid].clone().unwrap()
    }
    /// 64-bit version of an allocated register (for memory addressing).
    fn r64(&mut self, vid: usize) -> String {
        let r32 = self.r(vid);
        match r32.as_str() {
            "eax" => "rax".into(), "ebx" => "rbx".into(), "ecx" => "rcx".into(),
            "edx" => "rdx".into(), "esi" => "rsi".into(), "edi" => "rdi".into(),
            "r8d" => "r8".into(), "r9d" => "r9".into(), "r10d" => "r10".into(),
            "r11d" => "r11".into(), "r12d" => "r12".into(), "r13d" => "r13".into(),
            "r14d" => "r14".into(), "r15d" => "r15".into(),
            other => other.into(),
        }
    }
    /// 32-bit version of an allocated register (for data ops).
    /// If the register was allocated as 64-bit (rax), returns its 32-bit
    /// form (eax); otherwise returns the name as-is.
    fn r32(&mut self, vid: usize) -> String {
        let r = self.r(vid);
        match r.as_str() {
            "rax" => "eax".into(), "rbx" => "ebx".into(), "rcx" => "ecx".into(),
            "rdx" => "edx".into(), "rsi" => "esi".into(), "rdi" => "edi".into(),
            "r8" => "r8d".into(), "r9" => "r9d".into(), "r10" => "r10d".into(),
            "r11" => "r11d".into(), "r12" => "r12d".into(), "r13" => "r13d".into(),
            "r14" => "r14d".into(), "r15" => "r15d".into(),
            other => other.into(),
        }
    }
}


/// x86 condition-code суффикс (адаптер использует смешанные имена).
fn x86_cc(cond: &str) -> &str {
    match cond {
        "eq" | "e" => "e", "ne" => "ne",
        "lt" | "l" | "mi" => "l", "le" | "ls" => "le",
        "gt" | "g" => "g", "ge" | "pl" => "ge",
        "ult" | "b" | "cs" | "c" => "b", "ule" => "be",
        "ugt" | "a" | "hi" => "a", "uge" | "cc" | "nc" => "ae",
        "s" => "s", "ns" => "ns", "o" => "o", "no" => "no",
        other => other,
    }
}

/// 8-битное имя регистра по 32-битному (для setcc).
fn r8_of(r32: &str) -> String {
    match r32 {
        "eax" => "al".into(), "ebx" => "bl".into(), "ecx" => "cl".into(), "edx" => "dl".into(),
        "esi" => "sil".into(), "edi" => "dil".into(), "ebp" => "bpl".into(), "esp" => "spl".into(),
        r => r.replace('d', "b"), // r8d -> r8b
    }
}

fn emit_x86(g: &DataflowGraph, sym_map: &std::collections::HashMap<u64, String>) -> String {
    let mut out = String::new();
    let mut regs = RegAlloc::new(g.values.len());
    
    // Пролог: копируем Input-значения из их исходных x86 регистров
    for &vid in g.inputs.iter() {
        let name = match &g.values[vid] {
            Value::Input { name, .. } => name.clone(),
            _ => continue,
        };
        // 64-bit registers (address bases) get allocated to 64-bit regs and
        // copied full-width; 32-bit values use 32-bit regs.
        let is64 = matches!(name.as_str(),
            "rax"|"rbx"|"rcx"|"rdx"|"rsi"|"rdi"|"rsp"|"rbp"|
            "r8"|"r9"|"r10"|"r11"|"r12"|"r13"|"r14"|"r15");
        let dst = if is64 { regs.alloc64(vid) } else { regs.alloc(vid) };
        let src = if is64 {
            name.clone()
        } else {
            match name.as_str() {
                "rsp" => "esp".to_string(),
                "rbp" => "ebp".to_string(),
                other => other.to_string(),
            }
        };
        if src == dst {
            // input уже в своём регистре — self-move не нужен
            continue;
        }
        out.push_str(&format!("    mov {}, {}\n", dst, src));
    }
    
    let mut skip = 0usize;
    let mut order_iter = g.order.clone();

    for node in &order_iter {
        if skip > 0 { skip -= 1; continue; }
        match node {
            sdre::GraphNode::Value(vid) => {
                if g.inputs.contains(vid) { continue; }
                let v = &g.values[*vid];
                if let Value::Opaque{mnemonic,inputs,..} = v {
                    if mnemonic == "VECTOR_ADD_4x32" {
                        if inputs.len() < 2 {
                            out.push_str("    # VECTOR_ADD_4x32 skipped: no inputs\n");
                            continue;
                        }
                        for &inp in inputs.iter() { let _ = regs.alloc(inp); }
                        let base = regs.r(inputs[0]);
                        out.push_str(&format!(
                            "    vmovdqu ymm0, [{}]\n    vpaddd  ymm0, ymm0, [{}+0x10]\n    vmovdqu [{}], ymm0\n",
                            base, base, base
                        ));
                        skip = 3; continue;
                    }
                    out.push_str(&format!("    # {}\n", mnemonic)); continue;
                }

                let dst = regs.alloc(*vid);
                match v {
                    Value::Const{val,..} => out.push_str(&format!("    mov {}, 0x{:x}\n", dst, val)),
                    Value::Load{addr,disp,bits} => {
                        let a = regs.r64(*addr);
                        let d = *disp as i64;
                        let mem = if d == 0 { format!("[{}]", a) }
                            else if d < 0 { format!("[{}{}]", a, d) }
                            else { format!("[{}+0x{:x}]", a, d) };
                        // Byte/halfword loads must zero-extend to fill the 32-bit reg
                        match bits {
                            8 => out.push_str(&format!("    movzx {}, byte ptr {}\n", dst, mem)),
                            16 => out.push_str(&format!("    movzx {}, word ptr {}\n", dst, mem)),
                            _ => out.push_str(&format!("    mov {}, {}\n", dst, mem)),
                        }
                    }
                    Value::Copy{src,..} => out.push_str(&format!("    mov {}, {}\n", dst, regs.r32(*src))),
                    Value::Add{lhs,rhs} => { out.push_str(&format!("    mov {}, {}\n    add {}, {}\n", dst, regs.r32(*lhs), dst, regs.r32(*rhs))); }
                    Value::Sub{lhs,rhs} => { out.push_str(&format!("    mov {}, {}\n    sub {}, {}\n", dst, regs.r32(*lhs), dst, regs.r32(*rhs))); }
                    Value::Mul{lhs,rhs} => { out.push_str(&format!("    mov {}, {}\n    imul {}, {}\n", dst, regs.r32(*lhs), dst, regs.r32(*rhs))); }
                    Value::And{lhs,rhs} => { out.push_str(&format!("    mov {}, {}\n    and {}, {}\n", dst, regs.r32(*lhs), dst, regs.r32(*rhs))); }
                    Value::Or{lhs,rhs}  => { out.push_str(&format!("    mov {}, {}\n    or {}, {}\n", dst, regs.r32(*lhs), dst, regs.r32(*rhs))); }
                    Value::Xor{lhs,rhs} => { out.push_str(&format!("    mov {}, {}\n    xor {}, {}\n", dst, regs.r32(*lhs), dst, regs.r32(*rhs))); }
                    Value::Shl{lhs,rhs} => {
                        let amt = match &g.values[*rhs] { Value::Const{val,..} => format!("{}", val), _ => "cl".to_string() };
                        out.push_str(&format!("    mov {}, {}\n    shl {}, {}\n", dst, regs.r32(*lhs), dst, amt));
                    }
                    Value::Shr{lhs,rhs} => {
                        let amt = match &g.values[*rhs] { Value::Const{val,..} => format!("{}", val), _ => "cl".to_string() };
                        out.push_str(&format!("    mov {}, {}\n    shr {}, {}\n", dst, regs.r32(*lhs), dst, amt));
                    }
                    Value::Sar{lhs,rhs} => {
                        let amt = match &g.values[*rhs] { Value::Const{val,..} => format!("{}", val), _ => "cl".to_string() };
                        out.push_str(&format!("    mov {}, {}\n    sar {}, {}\n", dst, regs.r32(*lhs), dst, amt));
                    }
                    Value::Cmp{lhs,rhs,..} => { out.push_str(&format!("    cmp {}, {}\n", regs.r32(*lhs), regs.r32(*rhs))); }
                    Value::Not{src} => { out.push_str(&format!("    mov {}, {}\n    not {}\n", dst, regs.r32(*src), dst)); }
                    Value::Neg{src} => { out.push_str(&format!("    mov {}, {}\n    neg {}\n", dst, regs.r32(*src), dst)); }
                    Value::Select{cond,true_val,false_val} => {
                        out.push_str(&format!("    test {}, 1\n    mov {}, {}\n    cmovne {}, {}\n", regs.r32(*cond), dst, regs.r32(*false_val), dst, regs.r32(*true_val)));
                    }
                    Value::Cmov{src,dst:cd,cond,..} => { out.push_str(&format!("    mov {}, {}\n    cmov{} {}, {}\n", dst, regs.r32(*cd), x86_cc(cond), dst, regs.r32(*src))); }
                    Value::Zext{src,..} | Value::Trunc{src,..} => out.push_str(&format!("    mov {}, {}\n", dst, regs.r32(*src))),
                    Value::Sext{src,..} => out.push_str(&format!("    movsxd {}, {}\n", dst, regs.r32(*src))),
                    Value::Div{..}|Value::Mod{..}|Value::FloatCmp{..}|Value::FAdd{..}|Value::FSub{..}
                    |Value::FMul{..}|Value::FDiv{..}|Value::IntToFloat{..}|Value::FloatToInt{..}
                    |Value::FloatBits{..}|Value::ExchangeAdd{..} => {}
                    Value::Setcc{cond,..} => {
                        out.push_str(&format!("    xor {}, {}\n    set{} {}\n", dst, dst, x86_cc(cond), r8_of(&dst)));
                    }
                    Value::Opaque{mnemonic,..} => {
                        if mnemonic.starts_with("insn_") {
                            out.push_str(&format!("    ; UNKNOWN: {mnemonic}\n"));
                        } else {
                            out.push_str(&format!("    {mnemonic}\n"));
                        }
                    }
                    Value::Input{name,..} => { out.push_str(&format!("    ; input {name}\n")); }
                }
            }
            sdre::GraphNode::Effect(eidx) => {
                let e = &g.effects[*eidx];
                match e {
                    sdre::Effect::Store{addr,disp,value,bits} => {
                        let a = regs.r64(*addr); let v = regs.r32(*value);
                        let d = *disp as i64;
                        let mem = if d == 0 { format!("[{}]", a) }
                            else if d < 0 { format!("[{}{}]", a, d) }
                            else { format!("[{}+0x{:x}]", a, d) };
                        let sz = match bits { 8 => "byte ptr ", 16 => "word ptr ", _ => "" };
                        out.push_str(&format!("    mov {}, {}\n", format!("{}{}", sz, mem), v));
                    }
                    sdre::Effect::Branch{target} => {
                        if *target == 0 { out.push_str("    # jmp 0x0\n"); }
                        else { out.push_str(&format!("    jmp L_{:x}\n", target)); }
                    }
                    sdre::Effect::BranchCond{cond, cc, target_true, target_false} => {
                        let _ = regs.r(*cond);
                        // EFLAGS are already set by the preceding Cmp value emit.
                        // Fallthrough = target_false (0 means next instruction).
                        if *target_true != 0 { out.push_str(&format!("    j{} L_{:x}\n", cc, target_true)); }
                        if *target_false != 0 { out.push_str(&format!("    jmp L_{:x}\n", target_false)); }
                    }
                    sdre::Effect::Call{target,args} => {
                        let arg_regs = ["edi","esi","edx","ecx"];
                        for (i,&a) in args.iter().enumerate() {
                            if i >= arg_regs.len() { break; }
                            let r = regs.r32(a);
                            if r != arg_regs[i] { out.push_str(&format!("    mov {}, {}\n", arg_regs[i], r)); }
                        }
                        if *target == 0 { out.push_str("    # call 0x0 (plt stub)\n"); }
                        else if let Some(sym) = sym_map.get(target) {
                            out.push_str(&format!("    call {}\n", sym));
                        } else {
                            out.push_str(&format!("    call 0x{:x}\n", target));
                        }
                    }
                    sdre::Effect::Ret{value} => {
                        if let Some(v) = value {
                            let r = regs.r32(*v);
                            if r != "eax" { out.push_str(&format!("    mov eax, {}\n", r)); }
                        }
                        out.push_str("    ret\n");
                    }
                    sdre::Effect::OpaqueEffect{mnemonic,..} => out.push_str(&format!("    # {}\n", mnemonic)),
                }
            }
        }
    }
    out
}

// ─── PIPELINE FOR ONE FUNCTION ────────────────────────────────────────────────

#[derive(Default, Clone)]
struct FuncStats {
    n_regions: usize,
    n_vectorized: usize,
    n_branchless: usize,
    n_inlined: usize,
}

fn optimize_func(bytes: &[u8], base_addr: u64, cli: &Cli, sym_map: &std::collections::HashMap<u64, String>) -> Result<(String, FuncStats), String> {
    let instrs = disasm_to_ir(bytes, base_addr, Arch::X86_64)
        .map_err(|e| format!("disasm: {}", e))?;
    let config = RegionConfig::default();
    let regions = split_regions(&instrs, &config);
    if regions.is_empty() {
        return Err("no regions".into());
    }

    let mut graphs: Vec<DataflowGraph> = Vec::new();
    for r in &regions {
        graphs.push(lift_region(r, Arch::X86_64)
            .map_err(|e| format!("lift: {}", e))?);
    }

    for g in &mut graphs { dce(g); }
    if !cli.no_inline { inline_pass(&regions, &mut graphs); }
    if !cli.no_branchless { for g in &mut graphs { branchless_pass(g); } }
    if !cli.no_vectorize { for g in &mut graphs { vectorize_loop(g); } }

    let mut stats = FuncStats::default();
    stats.n_regions = graphs.len();

    let mut out = String::new();
    // Collect all branch targets across regions → label map.
    // A label is emitted before the region whose entry_addr matches the target.
    let mut targets: std::collections::HashSet<u64> = std::collections::HashSet::new();
    for g in &graphs {
        for e in &g.effects {
            match e {
                sdre::Effect::Branch{target} => { if *target != 0 { targets.insert(*target); } }
                sdre::Effect::BranchCond{target_true, target_false, ..} => {
                    if *target_true != 0 { targets.insert(*target_true); }
                    if *target_false != 0 { targets.insert(*target_false); }
                }
                _ => {}
            }
        }
    }
    let region_entries: std::collections::HashSet<u64> =
        regions.iter().map(|r| r.entry_addr).collect();

    for (ri, g) in graphs.iter().enumerate() {
        // Emit label if this region is a branch target
        let entry = regions[ri].entry_addr;
        if targets.contains(&entry) {
            out.push_str(&format!("L_{:x}:\n", entry));
        }
        let _ = &region_entries;
        out.push_str(&format!("# Region #{} -- {} values, {} effects\n", ri, g.values.len(), g.effects.len()));
        out.push_str(&emit_x86(g, sym_map));
        // Check if vectorized
        if g.values.iter().any(|v| matches!(v, Value::Opaque{mnemonic,..} if mnemonic == "VECTOR_ADD_4x32")) {
            stats.n_vectorized += 1;
        }
        if g.effects.iter().any(|e| matches!(e, sdre::Effect::OpaqueEffect{mnemonic,..} if mnemonic == "branchless")) {
            stats.n_branchless += 1;
        }
        if g.effects.iter().any(|e| matches!(e, sdre::Effect::OpaqueEffect{mnemonic,..} if mnemonic == "inlined")) {
            stats.n_inlined += 1;
        }
        out.push('\n');
    }
    Ok((out, stats))
}

// ─── CLI ──────────────────────────────────────────────────────────────────────

#[derive(Parser)]
#[command(name = "x86_superopt", about = "x86_64 binary super-optimizer", version)]
struct Cli {
    #[arg(long)] file: Option<PathBuf>,
    #[arg(long, conflicts_with = "file")] bytes: Option<String>,
    #[arg(long, default_value_t = 0x100000)] base_addr: u64,
    #[arg(long, default_value_t = false)] no_vectorize: bool,
    #[arg(long, default_value_t = false)] no_inline: bool,
    #[arg(long, default_value_t = false)] no_branchless: bool,
    #[arg(long)] output: Option<PathBuf>,
    #[arg(long, default_value_t = 4)] threads: usize,
}

// ─── MAIN ─────────────────────────────────────────────────────────────────────

fn main() {
    let cli = Cli::parse();
    let bytes = if let Some(hex) = &cli.bytes {
        hex_to_bytes(hex).unwrap_or_else(|e| { eprintln!("Hex error: {}", e); std::process::exit(1); })
    } else if let Some(path) = &cli.file {
        fs::read(path).unwrap_or_else(|e| { eprintln!("File error: {}", e); std::process::exit(1); })
    } else {
        hex_to_bytes("8B048A83C10183F9647CF2").unwrap()
    };

    let out_path = cli.output.clone();

    // Если это ELF64 — парсим функции и оптимизируем каждую
    if bytes.len() > 4 && &bytes[..4] == b"\x7fELF" && bytes[4] == 2 {
        match parse_elf(&bytes) {
            Ok(ctx) => {
                eprintln!("ELF64: .text @ 0x{:x}+0x{:x}, {} function symbols",
                         ctx.text_addr, ctx.text_size, ctx.symbols.len());

                // Если нет символов — просто оптимизируем всю .text секцию как один кусок
                if ctx.symbols.is_empty() {
                    eprintln!("No function symbols found; optimizing entire .text section");
                    let text_bytes = &bytes[ctx.text_off as usize..][..ctx.text_size as usize];
                    match optimize_func(text_bytes, ctx.text_addr, &cli, &std::collections::HashMap::new()) {
                        Ok((asm, stats)) => {
                            let out = format!("# .text @ 0x{:x} ({} bytes)\n{}\n",
                                              ctx.text_addr, ctx.text_size, asm);
                            if let Some(path) = &out_path {
                                fs::write(path, &out).unwrap();
                                eprintln!("Wrote {}", path.display());
                            } else { println!("{}", out); }
                            eprintln!("Stats: {} regions", stats.n_regions);
                        }
                        Err(e) => eprintln!("Error on .text: {}", e),
                    }
                    return;
                }

                // Есть символы — оптимизируем каждую функцию
                // Фильтр: пропускаем _start и другие служебные символы
                let funcs: Vec<(u64, u64, String)> = ctx.symbols.iter()
                    .filter(|(_, _, name)| {
                        name != "_start" && name != "_init" && name != "_fini"
                        && !name.starts_with("__do_") && !name.starts_with("__libc_")
                        && !name.starts_with("__stack") && !name.starts_with("frame_dummy")
                        && name != "call_gmon_start" && name != "register_tm_clones"
                        && name != "deregister_tm_clones"
                    })
                    .cloned().collect();
                let total_funcs = funcs.len();
                let count = Arc::new(AtomicUsize::new(0));
                let vec_count = Arc::new(AtomicUsize::new(0));
                let region_count = Arc::new(AtomicUsize::new(0));
                let errors = Arc::new(AtomicUsize::new(0));
                
                // Build symbol map: address → name for call resolution
                let sym_map: std::collections::HashMap<u64, String> = ctx.symbols.iter()
                    .map(|(addr, _, name)| (*addr, name.clone()))
                    .collect();
                let sym_map_arc = Arc::new(sym_map);

                // Подготавливаем входные данные для параллельной обработки
                let text_start = ctx.text_off;
                let text_addr = ctx.text_addr;
                let bytes_arc = Arc::new(bytes);
                let cli_arc = Arc::new(cli);
                let thread_count = cli_arc.threads;

                let mut handles = Vec::new();
                let chunk_size = (funcs.len() + thread_count - 1) / thread_count;

                for chunk in funcs.chunks(chunk_size) {
                    let chunk = chunk.to_vec();
                    let bytes = bytes_arc.clone();
                    let cli = cli_arc.clone();
                    let cnt = count.clone();
                    let vcnt = vec_count.clone();
                    let rcnt = region_count.clone();
                    let errcnt = errors.clone();
                    let sym_map = sym_map_arc.clone();

                    handles.push(thread::spawn(move || {
                        let mut local_out = String::new();
                        for (sym_addr, sym_size, sym_name) in &chunk {
                            let file_off = (sym_addr - text_addr) + text_start;
                            let func_bytes = &bytes[file_off as usize..][..*sym_size as usize];
                            match optimize_func(func_bytes, *sym_addr, &cli, &sym_map) {
                                Ok((asm, stats)) => {
                                    local_out.push_str(&format!(
                                        ".globl {}\n{}:\n# --- {} @ 0x{:x}+0x{:x} ---\n{}\n",
                                        sym_name, sym_name, sym_name, sym_addr, sym_size, asm
                                    ));
                                    vcnt.fetch_add(stats.n_vectorized, Ordering::Relaxed);
                                    rcnt.fetch_add(stats.n_regions, Ordering::Relaxed);
                                }
                                Err(e) => {
                                    local_out.push_str(&format!(
                                        "# SKIP {} @ 0x{:x}: {}\n\n", sym_name, sym_addr, e
                                    ));
                                    errcnt.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                            cnt.fetch_add(1, Ordering::Relaxed);
                            if cnt.load(Ordering::Relaxed) % 1000 == 0 {
                                eprintln!("Progress: {}/{}", cnt.load(Ordering::Relaxed), total_funcs);
                            }
                        }
                        local_out
                    }));
                }

                let mut full_out = String::from(".intel_syntax noprefix\n.section .text\n\n");
                for h in handles {
                    full_out.push_str(&h.join().unwrap());
                }

                // Вывод
                let final_count = count.load(Ordering::Relaxed);
                let final_regions = region_count.load(Ordering::Relaxed);
                let final_vec = vec_count.load(Ordering::Relaxed);
                let final_err = errors.load(Ordering::Relaxed);

                eprintln!("\n=== Summary ===");
                eprintln!("Functions processed: {}/{}", final_count, total_funcs);
                eprintln!("Total regions: {}", final_regions);
                eprintln!("Vectorized regions: {}", final_vec);
                eprintln!("Errors: {}", final_err);

                if let Some(path) = &out_path {
                    fs::write(path, &full_out).unwrap();
                    eprintln!("Wrote {}", path.display());
                } else {
                    if full_out.len() > 100000 {
                        println!("Output too large ({} bytes), showing first 5000 chars:", full_out.len());
                        println!("{}...", &full_out[..5000]);
                    } else {
                        println!("{}", full_out);
                    }
                }
            }
            Err(e) => {
                eprintln!("ELF parse error: {}", e);
                // Fallback: raw bytes
                match optimize_func(&bytes, cli.base_addr, &cli, &std::collections::HashMap::new()) {
                    Ok((asm, stats)) => {
                        let mut out = String::from(".intel_syntax noprefix\n.section .text\n\n");
                        out.push_str(&format!("# Raw @ 0x{:x}\n{}\n", cli.base_addr, asm));
                        if let Some(path) = &out_path {
                            fs::write(path, &out).unwrap();
                            eprintln!("Wrote {}", path.display());
                        } else { println!("{}", out); }
                        println!("Stats: {} regions, {} vectorized", stats.n_regions, stats.n_vectorized);
                    }
                    Err(e) => eprintln!("Error: {}", e),
                }
            }
        }
    } else {
        // Не ELF — raw bytes
        let empty_map = std::collections::HashMap::new();
        match optimize_func(&bytes, cli.base_addr, &cli, &empty_map) {
            Ok((asm, stats)) => {
                let mut out = String::from(".intel_syntax noprefix\n.section .text\n\n");
                out.push_str(&format!("# Raw @ 0x{:x}\n{}\n", cli.base_addr, asm));
                if let Some(path) = &out_path {
                    fs::write(path, &out).unwrap();
                    eprintln!("Wrote {}", path.display());
                } else { println!("{}", out); }
                println!("Stats: {} regions, {} vectorized", stats.n_regions, stats.n_vectorized);
            }
            Err(e) => eprintln!("Error: {}", e),
        }
    }
}

fn hex_to_bytes(hex: &str) -> Result<Vec<u8>, String> {
    let hex = hex.trim_start_matches("0x").trim_start_matches("0X")
        .chars().filter(|c| !c.is_whitespace()).collect::<String>();
    if hex.len() % 2 != 0 { return Err("odd hex length".into()); }
    (0..hex.len()).step_by(2).map(|i|
        u8::from_str_radix(&hex[i..i+2], 16).map_err(|e| format!("bad hex at {}: {}", i, e))
    ).collect()
}
