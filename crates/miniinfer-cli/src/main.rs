fn main() {
    let mut args =  std::env::args().skip(1);

    match args.next().as_deref() {
        Some("run") => println!("miniinfer run: not implemented yet"),
        Some("inspect") => println!("miniinfer inspect: not implemented yet"),
        Some("bench") => println!("miniinfer bench: not implemented yet"),
        Some("--help") | Some("-h") | None => print_help(),
        Some(cmd) => {
            eprintln!("Unknown command: {cmd}");
            print_help();
            std::process::exit(2);
        }
    }
}

fn print_help() {
    println!(
        "Usage: miniinfer <command> [options]
Commands:
    run       Run inference on a model (not implemented yet)
    inspect   Inspect a model (not implemented yet)
    bench     Benchmark a model (not implemented yet)
Options:
    -h, --help    Show this help message"
    );
}
