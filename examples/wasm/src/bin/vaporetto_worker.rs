use gloo_worker::Registrable;
use vaporetto_wasm::VaporettoWorker;

fn main() {
    VaporettoWorker::registrar().register();
}
