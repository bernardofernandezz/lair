use std::io::{self, BufRead, IsTerminal, Write};

use anyhow::Result;

use crate::engine::{Engine, Msg, Params, Part};

pub fn run(engine: &mut Engine, model: &str, params: Params, system: Option<String>) -> Result<()> {
    let base: Vec<Msg> = system
        .map(|s| Msg { role: "system".into(), content: s })
        .into_iter()
        .collect();
    let mut history = base.clone();
    let (dim, reset) = if io::stdout().is_terminal() { ("\x1b[2m", "\x1b[0m") } else { ("", "") };
    println!("Chatting with {model}. /reset clears history, /exit quits.");

    let stdin = io::stdin();
    loop {
        print!("\n> ");
        io::stdout().flush()?;
        let mut line = String::new();
        if stdin.lock().read_line(&mut line)? == 0 {
            return Ok(());
        }
        match line.trim() {
            "" => continue,
            "/exit" | "/quit" => return Ok(()),
            "/reset" => {
                history = base.clone();
                println!("(history cleared)");
                continue;
            }
            text => history.push(Msg { role: "user".into(), content: text.into() }),
        }

        let mut reply = String::new();
        let mut in_thought = false;
        let res = engine.generate(model, &history, params, |part| {
            match part {
                Part::Thinking(t) => {
                    if !in_thought {
                        print!("{dim}");
                        in_thought = true;
                    }
                    print!("{t}");
                }
                Part::Answer(t) => {
                    if in_thought {
                        print!("{reset}\n\n");
                        in_thought = false;
                    }
                    print!("{t}");
                    reply.push_str(t);
                }
            }
            let _ = io::stdout().flush();
            true
        });
        if in_thought {
            print!("{reset}");
        }
        println!();
        match res {
            // Reasoning stays out of the history, as chat templates expect.
            Ok(_) => history.push(Msg { role: "assistant".into(), content: reply }),
            Err(e) => {
                eprintln!("error: {e:#}");
                history.pop();
            }
        }
    }
}
