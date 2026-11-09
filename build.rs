fn main() {
    // rust-embed bakes web/dist in at compile time; rebuild when the UI is rebuilt.
    println!("cargo:rerun-if-changed=web/dist");
}
