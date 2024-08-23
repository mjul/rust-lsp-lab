# Rust LSP Lab: a Clojure Language Server in Rust using Tower-LSP 
This project demonstrates a small language server for a Clojure-like language.

It consists of a simple TypeScript extension for VS Code that acts as a client to the language server and
a server is built in Rust. The server uses (Tokio) Tower and Tokio LSP for the scaffolding (`tower-lsp`) 
and Chumsky (`chumsky`) parser combinators for parsing the source code.

See instructions below for how to build and run the project in VS Code.  

# Implementation Notes

## The Source Code Lexer and Parser
The language is a simplified Clojure syntax defined in [src/chumsky.rs](/src/chumsky.rs) using the 
[Chumsky](https://github.com/zesterer/chumsky) parser combinator library. 

At its core, the server parses a source document in a traditional two-phased approach:

First, the lexer [`chumsky::lexer()`](src/chumsky.rs) extracts the semantically interesting tokens.
Note that these are not just what would be needed for a compiler, but also tokens that are interesting to the client
application (e.g. the VS Code editor). Similar to the .NET Roslyn compilers you keep tokens and their spans,
the position range in the source file, so that the lexer tokens and the AST can be linked back to the source code.

In the second phase, the [`chumsky::parser()`](src/chumsky.rs) extracts an AST from the token stream from the lexer
and computes the various views of the source AST that are needed, e.g. the syntax highlight information,
information about defined functions and variables for completion and code navigation etc.

### Notes on using Chumsky
While parser combinators are generally easy to work with they can be quite frustrating to write and debug since 
they allow ambiguities that tools like Lexx and Yacc or Antlr would identify. 
Chumsky has quite good documentation.

## Notes on the LSP Server
See (main.rs)[src/main.rs] for the top-level server structure.

The LSP is an extensive protocol. Initially the server declares its capabilities (a subset of the full protocol), 
see `main::initialize`. In the same file you find the high-level protocols function entry-points for these capabilities.

### semantic_token_full
This operation sends the "semantic" token information to VS Code which is then used for semantic syntax highlighting 
(see https://code.visualstudio.com/api/language-extensions/semantic-highlight-guide). 
The information is passed in `ImCompleteSemanticToken`.

## Other Thoughts
For the next lab, start with the official VS Code Extension code. `tower-lsp-boilerplate` is good for
inspiration but also has a lot of noise when developing new languages. The underlying Rust library, `tower-lsp`, may 
still be useful for quickly building the server.

# References
This project was based on this project https://github.com/IWANABETHATGUY/tower-lsp-boilerplate
It looks like the Nano Rust language defined in this project is just an example from the Chumsky project.

The following is the README from this Tower-LSP-boilerplate project:

# boilerplate for a  rust language server powered by `tower-lsp` 
## Introduction
This repo is a template for `tower-lsp`, a useful github project template which makes writing new language servers easier.
## Development using VSCode
1. `pnpm i`
2. `cargo build`
3. Open the project in VSCode: `code .`
4. In VSCode, press <kbd>F5</kbd> or change to the Debug panel and click <kbd>Launch Client</kbd>.
5. In the newly launched VSCode instance, open the file `examples/test.nrs` from this project.
6. If the LSP is working correctly you should see syntax highlighting and the features described below should work.
> **Note**  
> 
> If encountered errors like `Cannot find module '/xxx/xxx/dist/extension.js'`
> please try run command `tsc -b` manually, you could refer https://github.com/IWANABETHATGUY/tower-lsp-boilerplate/issues/6 for more details
## A valid program in nano rust 
```rust
fn factorial(x) {
    // Conditionals are supported!
    if x == 0 {
        1
    } else {
        x * factorial(x - 1)
    }
}

// The main function
fn main() {
    let three = 3;
    let meaning_of_life = three * 14 + 1;

    print("Hello, world!");
    print("The meaning of life is...");

    if meaning_of_life == 42 {
        print(meaning_of_life);
    } else {
        print("...something we cannot know");

        print("However, I can tell you that the factorial of 10 is...");
        // Function calling
        print(factorial(10));
    }
}
```
## Features
This repo use a language `nano rust` which first introduced by [ chumsky ](https://github.com/zesterer/chumsky/blob/master/examples/nano_rust.rs). Most common language feature has been implemented, you could preview via the video below.

- [x] InlayHint for LiteralType
![inlay hint](https://user-images.githubusercontent.com/17974631/156926412-c3823dac-664e-430e-96c1-c003a86eabb2.gif)

- [x] semantic token   
make sure your semantic token is enabled, you could enable your `semantic token` by
adding this line  to your `settings.json`
```json
{
 "editor.semanticHighlighting.enabled": true,
}
```
- [x] syntactic error diagnostic

https://user-images.githubusercontent.com/17974631/156926382-a1c4c911-7ea1-4d3a-8e08-3cf7271da170.mp4

- [x] code completion  

https://user-images.githubusercontent.com/17974631/156926355-010ef2cd-1d04-435b-bd1e-8b0dab9f44f1.mp4

- [x] go to definition  

https://user-images.githubusercontent.com/17974631/156926103-94d90bd3-f31c-44e7-a2ce-4ddfde89bc33.mp4

- [x] find reference

https://user-images.githubusercontent.com/17974631/157367235-7091a36c-631a-4347-9c1e-a3b78db81714.mp4

- [x] rename

https://user-images.githubusercontent.com/17974631/157367229-99903896-5583-4f67-a6da-1ae1cf206876.mp4







