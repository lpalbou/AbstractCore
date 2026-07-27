fn main() {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    std::process::exit(abstractcore_console::run_cli(&argv));
}
