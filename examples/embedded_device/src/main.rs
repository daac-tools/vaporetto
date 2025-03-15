#![feature(alloc_error_handler)]
#![no_std]
#![no_main]

extern crate alloc;

// core crate
use core::alloc::Layout;

// alloc crate
use alloc::string::String;

// devices
use alloc_cortex_m::CortexMHeap;
use cortex_m::asm;
use cortex_m_rt::entry;
use cortex_m_semihosting::hprintln;

// other crates
use vaporetto::{CharacterType, Predictor, Sentence};
use vaporetto_rules::{sentence_filters::KyteaWsConstFilter, SentenceFilter};

// panic behaviour
use panic_halt as _;

#[global_allocator]
static ALLOCATOR: CortexMHeap = CortexMHeap::empty();

const HEAP_SIZE: usize = 40 * 1024; // in bytes

#[entry]
fn main() -> ! {
    unsafe { ALLOCATOR.init(cortex_m_rt::heap_start() as usize, HEAP_SIZE) }

    let predictor_data = include_bytes!(concat!(env!("OUT_DIR"), "/predictor.bin"));
    let (predictor, _) =
        unsafe { Predictor::deserialize_from_slice_unchecked(predictor_data) }.unwrap();

    let docs =
        &["🚤VaporettoはSTM32F303VCT6(FLASH:256KiB,RAM:40KiB)などの小さなデバイスでも動作します"];

    let wsconst_d_filter = KyteaWsConstFilter::new(CharacterType::Digit);

    loop {
        for &text in docs {
            hprintln!("\x1b[32mINPUT:\x1b[m {:?}", text);
            let mut s = Sentence::from_raw(text).unwrap();
            predictor.predict(&mut s);
            wsconst_d_filter.filter(&mut s);
            let mut buf = String::new();
            s.write_tokenized_text(&mut buf);
            hprintln!("\x1b[31mOUTPUT:\x1b[m {}", buf);
        }
    }
}

#[alloc_error_handler]
fn alloc_error(_layout: Layout) -> ! {
    hprintln!("alloc error");
    asm::bkpt();

    loop {}
}
