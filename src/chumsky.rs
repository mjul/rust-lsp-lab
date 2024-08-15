//! This is the parser and interpreter for a simplified Clojure grammar
//! See the complete ANTLR 4 grammar here:
//! https://github.com/antlr/grammars-v4/blob/master/clojure/Clojure.g4

use core::fmt;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use chumsky::Parser;
use chumsky::{prelude::*, stream::Stream};
use serde::{Deserialize, Serialize};
use tower_lsp::lsp_types::SemanticTokenType;

use crate::semantic_token::LEGEND_TYPE;

pub type Span = std::ops::Range<usize>;

/// Semantic Tokens are used by clients for syntax highlighting.
/// See https://microsoft.github.io/language-server-protocol/specifications/lsp/3.17/specification/#textDocument_semanticTokens
#[derive(Debug)]
pub struct ImCompleteSemanticToken {
    pub start: usize,
    pub length: usize,
    pub token_type: usize,
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct NameToken(String);

impl std::fmt::Display for NameToken {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SymbolToken {
    Dot,
    Slash,
    Name(String),
}

impl std::fmt::Display for SymbolToken {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolToken::Dot => write!(f, "."),
            SymbolToken::Slash => write!(f, "/"),
            SymbolToken::Name(n) => write!(f, "{}", n),
        }
    }
}

// Note this is a simplified lexer
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LexerToken {
    String(String),
    Long(i64),
    Nil,
    Boolean(bool),
    Symbol(SymbolToken),
    NsSymbol(NameToken, SymbolToken),
    Character(char),
    Colon,
    /// Left parenthesis, '('
    LPar,
    /// Right parenthesis, ')'
    RPar,
}

impl Display for LexerToken {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LexerToken::Character(c) => write!(f, "'{}'", c),
            LexerToken::String(s) => write!(f, "\"{}\"", s),
            LexerToken::Long(n) => write!(f, "{}", n),
            LexerToken::Nil => write!(f, "nil"),
            LexerToken::Boolean(b) => write!(
                f,
                "{}",
                match b {
                    true => "true",
                    false => "false",
                }
            ),
            LexerToken::Symbol(s) => write!(f, "{}", s),
            LexerToken::NsSymbol(n, s) => write!(f, "{}/{}", n, s),
            LexerToken::Colon => write!(f, ":"),
            LexerToken::LPar => write!(f, "("),
            LexerToken::RPar => write!(f, ")"),
        }
    }
}
// TODO: continue here

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SymbolLiteral {
    NsSymbol(NameToken, SymbolToken),
    SimpleSymbol(SymbolToken),
}

impl fmt::Display for SymbolLiteral {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SymbolLiteral::NsSymbol(ns, s) => write!(f, ":{}/{}", ns, s),
            SymbolLiteral::SimpleSymbol(s) => write!(f, ":{}", s),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum KeywordLiteral {
    SimpleKeyword(SymbolLiteral),
    MacroKeyword(SymbolLiteral),
}

impl fmt::Display for KeywordLiteral {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            KeywordLiteral::SimpleKeyword(st) => write!(f, ":{}", st),
            KeywordLiteral::MacroKeyword(st) => write!(f, "::{}", st),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum LiteralExpr {
    Nil,
    Str(String),
    Number(String),
    Character(char),
    Bool(bool),
    Keyword(KeywordLiteral),
    Symbol(String),
}

impl fmt::Display for LiteralExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            LiteralExpr::Nil => write!(f, "nil"),
            LiteralExpr::Bool(x) => write!(f, "{}", x),
            LiteralExpr::Number(n) => write!(f, "{}", n),
            LiteralExpr::Str(s) => write!(f, "{}", s),
            LiteralExpr::Character(c) => write!(f, "{}", c),
            LiteralExpr::Keyword(kwt) => write!(f, "{}", kwt),
            LiteralExpr::Symbol(st) => write!(f, "{}", st),
        }
    }
}

fn lexer() -> impl Parser<char, Vec<(LexerToken, Span)>, Error = Simple<char>> {
    // A parser for comments
    let comment = just(";")
        .then(take_until(text::newline::<Simple<char>>()))
        .padded()
        .labelled("comment");

    // A parser for simple strings
    let str_ = just::<_, _, Simple<char>>('"')
        .ignore_then(filter(|c| *c != '"').repeated())
        .then_ignore(just('"'))
        .collect::<String>()
        .map(LexerToken::String)
        .labelled("string");

    // A parser for longs
    let long_number = text::int::<_, Simple<char>>(10)
        .map(|s| LexerToken::Long(s.parse().unwrap()))
        .labelled("LONG");
    // TODO: negative numbers
    // TODO: floats
    // TODO: floats, other bases
    /*
    let float_number = text::int(10)
        .chain::<char, _, _>(just('.').chain(text::digits(10)).or_not().flatten())
        .collect::<String>()
        .map(|s| LexerToken::Float(Float::parse(s)));
    */
    let number = long_number.labelled("number");

    // A parser for characters (simplified)
    let char_ = just::<_, _, Simple<char>>('\'')
        .ignore_then(none_of("'"))
        .then_ignore(just('\''))
        .map(|c| LexerToken::Character(c))
        .labelled("character");

    // TODO: use whitespace lexer for last part:
    let symbol_head = none_of::<_, _, Simple<char>>("0123456789^`\\\"#~@:/%()[]{} \n\r\t")
        .labelled("SYMBOL_HEAD fragment");
    let symbol_rest = choice::<_, Simple<char>>((
        symbol_head.clone(),
        one_of::<_, _, Simple<char>>("0123456789"),
        just::<_, _, Simple<char>>('.'),
    ))
    .labelled("SYMBOL_REST fragment");

    let name = symbol_head
        .then(symbol_rest.clone().repeated())
        .then(
            just::<_, _, Simple<char>>(':')
                .then(symbol_rest.repeated().at_least(1))
                .repeated(),
        )
        .map(|((sh, srs), colon_rests)| {
            let mut s = String::new();
            s.push(sh);
            for x in srs {
                s.push(x);
            }
            for (colon, symbol_rests) in colon_rests {
                s.push(colon);
                for x in symbol_rests {
                    s.push(x);
                }
            }
            NameToken(s)
        })
        .labelled("NAME fragment");

    let symbol_token = choice::<_, Simple<char>>((
        just::<_, _, Simple<char>>('.').to(SymbolToken::Dot),
        just::<_, _, Simple<char>>('/').to(SymbolToken::Slash),
        name.clone().map(|NameToken(s)| SymbolToken::Name(s)),
    ))
    .labelled("SYMBOL");

    let symbol = symbol_token
        .clone()
        .map(|st| LexerToken::Symbol(st))
        .labelled("simple_symbol");

    let ns_symbol = name
        .then(just('/'))
        .then(symbol_token)
        .map(|((n, _slash), symbol)| LexerToken::NsSymbol(n, symbol))
        .labelled("ns_symbol");

    /*
            let simple_keyword_ = just(':')
                .ignore_then(symbol_)
                .collect::<String>()
                .map(KeywordLiteral::SimpleKeyword);

            let ns_keyword_ = just(':')
                .ignore_then(symbol_)
                .then(just('/'))
                .ignore_then(symbol_)
                .collect::<String>()
                .map(_ => KeywordLiteral::MacroKeyword("x", ))
    */

    // A parser for keywords
    /*
        let keyword_ = just(':')
            .then(just(':'))
            .ignore_then(filter(|c| *c != '"').repeated())
            .then_ignore(just('"'))
            .collect::<String>()
            .map(Literal::Number);
    */

    // A parser for control characters (delimiters, semicolons, etc.)
    //let ctrl = one_of("()[]{};,").map(Literal::Ctrl);

    let lpar = just::<_, _, Simple<char>>('(').to(LexerToken::LPar);
    let rpar = just::<_, _, Simple<char>>(')').to(LexerToken::RPar);
    let colon = just::<_, _, Simple<char>>(':').to(LexerToken::Colon);
    let boolean_true =
        text::keyword::<_, _, Simple<char>>("true").map(|_| LexerToken::Boolean(true));
    let boolean_false = text::keyword("false").map(|_| LexerToken::Boolean(false));
    let boolean = boolean_true.or(boolean_false).labelled("BOOLEAN");

    let nil = text::keyword::<_, _, Simple<char>>("nil")
        .map(|_| LexerToken::Nil)
        .labelled("NIL");

    // A single token can be one of the above
    let token = choice((
        lpar, rpar, colon, char_, str_, boolean, nil, number, ns_symbol, symbol,
    ))
    .recover_with(skip_then_retry_until([]));

    token
        .map_with_span(|tok, span| {
            println!("{:?} {:?}", &tok, &span);
            (tok, span)
        })
        .padded()
        .padded_by(comment.ignored().repeated())
        .repeated()
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    Num(f64),
    Str(String),
    List(Vec<Value>),
    Func(String),
}

impl std::fmt::Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            Self::Null => write!(f, "null"),
            Self::Bool(x) => write!(f, "{}", x),
            Self::Num(x) => write!(f, "{}", x),
            Self::Str(x) => write!(f, "{}", x),
            Self::List(xs) => write!(
                f,
                "[{}]",
                xs.iter()
                    .map(|x| x.to_string())
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
            Self::Func(name) => write!(f, "<function: {}>", name),
        }
    }
}

#[deprecated]
#[derive(Clone, Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    NotEq,
}

pub type Spanned<T> = (T, Span);

/// The top-most AST expression, representing a complete source file, a sequence of `FormExpr`.
/// ```EBNF
///   file : form* ;
/// ```
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub enum FileExpr {
    Forms(Vec<Spanned<FormExpr>>),
}

#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub enum FormExpr {
    Literal(Box<Spanned<LiteralExpr>>),
    List(Box<Spanned<ListExpr>>),
    Vector(Box<VectorExpr>),
    Map(Box<MapExpr>),
    // TODO: set
    // No reader macros
    // ReaderMacro(ReaderMacroExpr)
}

impl FormExpr {
    /// Is this a literal expression?
    pub fn is_literal(&self) -> bool {
        matches!(self, Self::Literal(..))
    }
    /// Is this a list expression?
    pub fn is_list(&self) -> bool {
        matches!(self, Self::List(..))
    }
    /// Is this a vector expression?
    pub fn is_vector(&self) -> bool {
        matches!(self, Self::Vector(..))
    }
    /// Is this a map expression?
    pub fn is_map(&self) -> bool {
        matches!(self, Self::Map(..))
    }
}

/// AST expression for a sequence of forms (`FormExpr`).
/// ```EBNF
///   form = forms* ;
/// ```
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub struct FormsExpr(Box<Vec<Spanned<FormExpr>>>);

/// List expression:
/// ```EBNF
///   list : '(' forms ')' ;
/// ```
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub struct ListExpr(Spanned<FormsExpr>);

/// Vector expression:
/// ```EBNF
///   vector : '[' forms ']' ;
/// ```
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub struct VectorExpr(Spanned<FormsExpr>);

/// Map expression:
/// ```EBNF
///   map : '{' (form form)* '}' ;
/// ```
#[derive(Clone, Debug, PartialEq, Hash, Eq)]
pub struct MapExpr(Spanned<FormsExpr>);

// TODO: remove this (legacy)
// An expression node in the AST. Children are spanned so we can generate useful runtime errors.
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    Error,
    Value(Value),
    List(Vec<Spanned<Self>>),
    Local(Spanned<String>),
}

#[allow(unused)]
impl Expr {
    /// Returns `true` if the expr is [`Error`].
    ///
    /// [`Error`]: Expr::Error
    fn is_error(&self) -> bool {
        matches!(self, Self::Error)
    }

    /// Returns `true` if the expr is [`Value`].
    ///
    /// [`Value`]: Expr::Value
    fn is_value(&self) -> bool {
        matches!(self, Self::Value(..))
    }

    fn try_into_value(self) -> Result<Value, Self> {
        if let Self::Value(v) = self {
            Ok(v)
        } else {
            Err(self)
        }
    }

    fn as_value(&self) -> Option<&Value> {
        if let Self::Value(v) = self {
            Some(v)
        } else {
            None
        }
    }
}

// A function node in the AST.
#[derive(Clone, Debug, PartialEq)]
pub struct Func {
    pub args: Vec<Spanned<String>>,
    pub body: Spanned<Expr>,
    pub name: Spanned<String>,
    pub span: Span,
}

/*
fn lexer_to_literal_expr(input: &str) -> impl Parser<char, Spanned<LiteralExpr>, Error=Simple<char>>  {
    let (tokens, errors) = lexer().parse_recovery(input);
    let exprs: Vec<Spanned<LiteralExpr>> = match tokens {
        Some(token_spans) => {
            token_spans.into_iter().map(|(t, s)| {
                let expr = match t {
                    LexerToken::String(s) => LiteralExpr::Str(s),
                    LexerToken::Long(_) => todo!(),
                    LexerToken::Nil => LiteralExpr::Nil,
                    LexerToken::Boolean(x) => LiteralExpr::Bool(x),
                    LexerToken::Symbol(s) => LiteralExpr::Symbol(s.to_string()),
                    LexerToken::NsSymbol(ns, s) => LiteralExpr::Symbol(format!("{}/{}", ns.to_string(), s.to_string()))
                };
                Spanned::new(expr, s)
            })
                .collect()
        }
        None => vec![]
    };

    let parse_errors = vec![];
    let semantic_tokens = vec![];
    let ast = None;
    ParserResult { ast, parse_errors, semantic_tokens, }
}
*/

/// Parse lexer tokens to a literal expression, `LiteralExpr`
fn literal_expr_parser() -> impl Parser<LexerToken, Spanned<LiteralExpr>, Error = Simple<LexerToken>>
{
    let keyword = just(LexerToken::Colon)
        .repeated()
        .at_least(1)
        .at_most(2)
        .then(filter(|t| match t {
            LexerToken::Symbol(_) => true,
            _ => false,
        }))
        .map_with_span(|(colons, symbol), span| match symbol {
            LexerToken::Symbol(s) => Spanned::new(
                match colons.len() {
                    1 => LiteralExpr::Keyword(KeywordLiteral::SimpleKeyword(
                        SymbolLiteral::SimpleSymbol(s),
                    )),
                    2 => LiteralExpr::Keyword(KeywordLiteral::MacroKeyword(
                        SymbolLiteral::SimpleSymbol(s),
                    )),
                    _ => unreachable!(),
                },
                span,
            ),
            _ => unreachable!(),
        });

    let simple_literal = none_of::<LexerToken, _, _>(vec![LexerToken::LPar, LexerToken::RPar])
        .map_with_span(|t, span| match t {
            LexerToken::Nil => Spanned::new(LiteralExpr::Nil, span),
            LexerToken::Character(c) => Spanned::new(LiteralExpr::Character(c), span),
            LexerToken::String(s) => Spanned::new(LiteralExpr::Str(s), span),
            LexerToken::Long(n) => Spanned::new(LiteralExpr::Number(format!("{}", n)), span),
            LexerToken::Boolean(b) => Spanned::new(LiteralExpr::Bool(b), span),
            LexerToken::Symbol(s) => Spanned::new(LiteralExpr::Symbol(s.to_string()), span),
            LexerToken::NsSymbol(n, s) => Spanned::new(
                LiteralExpr::Symbol(format!("{}/{}", n.to_string(), s.to_string())),
                span,
            ),
            LexerToken::Colon => unreachable!(),
            LexerToken::LPar => unreachable!(),
            LexerToken::RPar => unreachable!(),
        });

    choice((keyword, simple_literal))
}

/// Parse lexer tokens to a form expression, `FormExpr`
fn form_expr_parser() -> impl Parser<LexerToken, Spanned<FormExpr>, Error = Simple<LexerToken>> {
    recursive(|form_expr_parser| {
        let forms = form_expr_parser
            .repeated()
            .map_with_span(|forms, span| (FormsExpr(Box::new(forms)), span))
            .labelled("forms");

        let literal = literal_expr_parser()
            .map(|(lit, span)| {
                Spanned::new(
                    FormExpr::Literal(Box::new(Spanned::new(lit, span.clone()))),
                    span,
                )
            })
            .labelled("literal");

        let list = forms
            .delimited_by::<_, _, _, _>(just(LexerToken::LPar), just(LexerToken::RPar))
            .map_with_span(|forms_expr, span: Span| {
                Spanned::new(
                    FormExpr::List(Box::new(Spanned::new(ListExpr(forms_expr), span.clone()))),
                    span.clone(),
                )
            })
            .labelled("list");

        // TODO: add literal, vector, etc.
        choice((literal, list)).labelled("form")
    })
}

// TODO: change to Expr
fn expr_parser() -> impl Parser<LiteralExpr, Spanned<Expr>, Error = Simple<LiteralExpr>> + Clone {
    recursive(|expr| {
        let raw_expr = recursive(|raw_expr| {
            let val = filter_map(|span, tok| match tok {
                LiteralExpr::Nil => Ok(Expr::Value(Value::Null)),
                LiteralExpr::Bool(x) => Ok(Expr::Value(Value::Bool(x))),
                LiteralExpr::Number(n) => Ok(Expr::Value(Value::Num(n.parse().unwrap()))),
                LiteralExpr::Str(s) => Ok(Expr::Value(Value::Str(s))),
                _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
            })
            .labelled("literal");

            /*
            let symbol = filter_map(|span, tok| match tok {
                Literal::Symbol(s) => Ok((s, span)),
                _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
            })
                .labelled("symbol");
                */

            /*

                        // A list of expressions
                        let items = expr
                            .clone()
                            .chain(just(Literal::Ctrl(',')).ignore_then(expr.clone()).repeated())
                            .then_ignore(just(Literal::Ctrl(',')).or_not())
                            .or_not()
                            .map(|item| item.unwrap_or_default());

                        let list = items
                            .clone()
                            .delimited_by(just(Literal::Ctrl('[')), just(Literal::Ctrl(']')))
                            .map(Expr::List);

                        // 'Atoms' are expressions that contain no ambiguity
                        let atom = val
                            .or(ident.map(Expr::Local))
                            .or(let_)
                            .or(list)
                            // In Nano Rust, `print` is just a keyword, just like Python 2, for simplicity
                            .or(just(Literal::Print)
                                .ignore_then(
                                    expr.clone()
                                        .delimited_by(just(Literal::Ctrl('(')), just(Literal::Ctrl(')'))),
                                )
                                .map(|expr| Expr::Print(Box::new(expr))))
                            .map_with_span(|expr, span| (expr, span))
                            // Atoms can also just be normal expressions, but surrounded with parentheses
                            .or(expr
                                .clone()
                                .delimited_by(just(Literal::Ctrl('(')), just(Literal::Ctrl(')'))))
                            // Attempt to recover anything that looks like a parenthesised expression but contains errors
                            .recover_with(nested_delimiters(
                                Literal::Ctrl('('),
                                Literal::Ctrl(')'),
                                [
                                    (Literal::Ctrl('['), Literal::Ctrl(']')),
                                    (Literal::Ctrl('{'), Literal::Ctrl('}')),
                                ],
                                |span| (Expr::Error, span),
                            ))
                            // Attempt to recover anything that looks like a list but contains errors
                            .recover_with(nested_delimiters(
                                Literal::Ctrl('['),
                                Literal::Ctrl(']'),
                                [
                                    (Literal::Ctrl('('), Literal::Ctrl(')')),
                                    (Literal::Ctrl('{'), Literal::Ctrl('}')),
                                ],
                                |span| (Expr::Error, span),
                            ));
            */

            val.clone().map_with_span(|expr, span| (expr, span))
        });

        raw_expr

        // Blocks are expressions but delimited with braces
        /*
        let block = expr
            .clone()
            .delimited_by(just(Literal::Ctrl('{')), just(Literal::Ctrl('}')))
            // Attempt to recover anything that looks like a block but contains errors
            .recover_with(nested_delimiters(
                Literal::Ctrl('{'),
                Literal::Ctrl('}'),
                [
                    (Literal::Ctrl('('), Literal::Ctrl(')')),
                    (Literal::Ctrl('['), Literal::Ctrl(']')),
                ],
                |span| (Expr::Error, span),
            ));
        */
    })
}

pub fn funcs_parser(
) -> impl Parser<LexerToken, HashMap<String, Func>, Error = Simple<LexerToken>> + Clone {
    let ident = filter_map(|span, tok| match tok {
        //Literal::Ident(ident) => Ok(ident),
        _ => Err(Simple::expected_input_found(span, Vec::new(), Some(tok))),
    });

    ident.then_ignore(end())

    // Argument lists are just identifiers separated by commas, surrounded by parentheses
    /*
        let args = ident
            .map_with_span(|name, span| (name, span))
            .separated_by(just(Literal::Ctrl(',')))
            .allow_trailing()
            .delimited_by(just(Literal::Ctrl('(')), just(Literal::Ctrl(')')))
            .labelled("function args");
    */
    /*
    let func = just(Literal::Fn)
        .ignore_then(
            ident
                .map_with_span(|name, span| (name, span))
                .labelled("function name"),
        )
        .then(args)
        .then(
            expr_parser()
                .delimited_by(just(Literal::Ctrl('{')), just(Literal::Ctrl('}')))
                // Attempt to recover anything that looks like a function body but contains errors
                .recover_with(nested_delimiters(
                    Literal::Ctrl('{'),
                    Literal::Ctrl('}'),
                    [
                        (Literal::Ctrl('('), Literal::Ctrl(')')),
                        (Literal::Ctrl('['), Literal::Ctrl(']')),
                    ],
                    |span| (Expr::Error, span),
                )),
        )
        .map_with_span(|((name, args), body), span| {
            (
                name.clone(),
                Func {
                    args,
                    body,
                    name,
                    span,
                },
            )
        })
        .labelled("function");

    func.repeated()
        .try_map(|fs, _| {
            let mut funcs = HashMap::new();
            for ((name, name_span), f) in fs {
                if funcs.insert(name.clone(), f).is_some() {
                    return Err(Simple::custom(
                        name_span,
                        format!("Function '{}' already exists", name),
                    ));
                }
            }
            Ok(funcs)
        })
        .then_ignore(end())
        */
}

pub fn type_inference(expr: &Spanned<Expr>, symbol_type_table: &mut HashMap<Span, Value>) {
    match &expr.0 {
        Expr::Error => {}
        Expr::Value(_) => {}
        Expr::List(exprs) => exprs
            .iter()
            .for_each(|expr| type_inference(expr, symbol_type_table)),
        Expr::Local(_) => {}
    }
}

#[derive(Debug)]
pub struct ParserResult {
    pub ast: Option<HashMap<String, Func>>,
    pub parse_errors: Vec<Simple<String>>,
    pub semantic_tokens: Vec<ImCompleteSemanticToken>,
}

/// Parse an input to a `ParseResult` with the semantic information for syntax highlighting
pub fn parse(src: &str) -> ParserResult {
    // First, the lexing
    let (tokens, errs) = lexer().parse_recovery(src);

    log::debug!("Lexer Tokens: {:?}", tokens);
    log::debug!("Lexer Errors: {:?}", errs);

    let (ast, tokenize_errors, semantic_tokens) = if let Some(tokens) = tokens {
        // First we collect the semantic tokens for syntax highlighting from the lexer tokens
        // info!("Tokens = {:?}", tokens);
        let semantic_tokens = tokens
            .iter()
            .filter_map(|(token, span)| match token {
                LexerToken::Colon => None,
                LexerToken::LPar => None,
                LexerToken::RPar => None,
                LexerToken::Nil | LexerToken::Boolean(_) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::KEYWORD)
                        .unwrap(),
                }),
                LexerToken::Character(_) | LexerToken::String(_) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::STRING)
                        .unwrap(),
                }),
                LexerToken::Long(_) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::NUMBER)
                        .unwrap(),
                }),
                LexerToken::Symbol(_) | LexerToken::NsSymbol(_, _) => {
                    Some(ImCompleteSemanticToken {
                        start: span.start,
                        length: span.len(),
                        token_type: LEGEND_TYPE
                            .iter()
                            .position(|item| item == &SemanticTokenType::VARIABLE)
                            .unwrap(),
                    })
                }
            })
            .collect::<Vec<_>>();

        let len = src.chars().count();
        // Now parse the lexemes (tokens) into an AST
        let (ast, parse_errs) =
            funcs_parser().parse_recovery(Stream::from_iter(len..len + 1, tokens.into_iter()));

        (ast, parse_errs, semantic_tokens)
    } else {
        (None, Vec::new(), vec![])
    };

    let parse_errors = errs
        .into_iter()
        .map(|e| e.map(|c| c.to_string()))
        .chain(
            tokenize_errors
                .into_iter()
                .map(|e| e.map(|tok| tok.to_string())),
        )
        .collect::<Vec<_>>();

    ParserResult {
        ast,
        parse_errors,
        semantic_tokens,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod lexing {
        use super::*;

        fn lex_single(input: &str) -> (LexerToken, Span) {
            let actual = lexer().parse(input);
            assert!(actual.is_ok());
            let tokens = actual.unwrap();
            assert_eq!(1, tokens.len());
            let (t, s) = tokens.first().unwrap();
            (t.clone(), s.clone())
        }

        #[test]
        fn lexer_can_parse_long() {
            let (lt, span) = lex_single("42");
            assert_eq!(LexerToken::Long(42), lt);
            assert_eq!(2, span.len());
            assert_eq!(0, span.start);
            assert_eq!(2, span.end);
        }

        macro_rules! can_parse {
            ($tc_name:ident, $input:literal, $expected:expr) => {
                #[test]
                fn $tc_name() {
                    let (t, _span) = lex_single($input);
                    assert_eq!($expected, t);
                }
            };
        }

        can_parse!(colon, ":", LexerToken::Colon);
        can_parse!(left_parenthesis, "(", LexerToken::LPar);
        can_parse!(right_parenthesis, ")", LexerToken::RPar);
        can_parse!(nil, "nil", LexerToken::Nil);
        can_parse!(boolean_true, "true", LexerToken::Boolean(true));
        can_parse!(boolean_false, "false", LexerToken::Boolean(false));
        can_parse!(char_simple, "'x'", LexerToken::Character('x'));

        can_parse!(string, "\"foo\"", LexerToken::String(String::from("foo")));

        can_parse!(symbol_dot, ".", LexerToken::Symbol(SymbolToken::Dot));
        can_parse!(symbol_slash, "/", LexerToken::Symbol(SymbolToken::Slash));
        can_parse!(
            symbol_name_simple_single_letter,
            "x",
            LexerToken::Symbol(SymbolToken::Name(String::from("x")))
        );
        can_parse!(
            symbol_name_simple_word,
            "foo",
            LexerToken::Symbol(SymbolToken::Name(String::from("foo")))
        );
        can_parse!(
            symbol_name_simple_word_mixed,
            "x123",
            LexerToken::Symbol(SymbolToken::Name(String::from("x123")))
        );
        can_parse!(
            symbol_name_simple_dashed,
            "foo-bar",
            LexerToken::Symbol(SymbolToken::Name(String::from("foo-bar")))
        );

        can_parse!(
            symbol_with_namespace_short,
            "f/foo",
            LexerToken::NsSymbol(
                NameToken(String::from("f")),
                SymbolToken::Name(String::from("foo")),
            )
        );
        can_parse!(
            symbol_with_namespace_dotted,
            "name.space/foo",
            LexerToken::NsSymbol(
                NameToken(String::from("name.space")),
                SymbolToken::Name(String::from("foo")),
            )
        );

        #[test]
        fn lexer_can_parse_multiple_tokens_two_true_nil() {
            let actual = lexer().parse("true nil");
            assert_eq!(true, actual.is_ok());
            let tokens = actual.unwrap();
            assert_eq!(2, tokens.len());
            match &tokens[..] {
                [(t1, s1), (t2, s2)] => {
                    assert_eq!(&LexerToken::Boolean(true), t1);
                    assert_eq!(&LexerToken::Nil, t2);
                }
                _ => assert!(false),
            }
        }
        #[test]
        fn lexer_can_parse_multiple_tokens_two_with_comment() {
            let source = "1\n;; comment\n2";
            let actual = lexer().parse(source);
            assert_eq!(true, actual.is_ok());
            let tokens = actual.unwrap();
            assert_eq!(2, tokens.len());
            match &tokens[..] {
                [(t1, s1), (t2, s2)] => {
                    assert_eq!(&LexerToken::Long(1), t1);
                    assert_eq!("1", &source[Span::from(s1.clone())]);
                    assert_eq!(0..1, *s1);
                    assert_eq!(&LexerToken::Long(2), t2);
                    assert_eq!("2", &source[Span::from(s2.clone())]);
                    assert_eq!(13..14, *s2);
                }
                _ => assert!(false),
            }
        }

        #[test]
        fn lexer_can_parse_multiple_tokens_many() {
            let actual = lexer().parse(
                r#"
            ;;comment
            nil
            true
            false
            1
            "foo-string"
            .
            /
            foo
            foo.bar/baz
            "#,
            );
            assert_eq!(true, actual.is_ok());
            let tokens = actual.unwrap();
            assert_eq!(9, tokens.len());
            match &tokens[..] {
                [(t1, _), (t2, _), (t3, _), (t4, _), (t5, _), (t6, _), (t7, _), (t8, _), (t9, _)] =>
                {
                    assert_eq!(&LexerToken::Nil, t1);
                    assert_eq!(&LexerToken::Boolean(true), t2);
                    assert_eq!(&LexerToken::Boolean(false), t3);
                    assert_eq!(&LexerToken::Long(1), t4);
                    assert_eq!(&LexerToken::String("foo-string".to_string()), t5);
                    assert_eq!(&LexerToken::Symbol(SymbolToken::Dot), t6);
                    assert_eq!(&LexerToken::Symbol(SymbolToken::Slash), t7);
                    assert_eq!(
                        &LexerToken::Symbol(SymbolToken::Name(String::from("foo"))),
                        t8
                    );
                    assert_eq!(
                        &LexerToken::NsSymbol(
                            NameToken(String::from("foo.bar")),
                            SymbolToken::Name(String::from("baz")),
                        ),
                        t9
                    );
                }
                _ => assert!(false),
            }
        }

        #[test]
        fn lexer_can_parse_code_snippet_ns_form_simple() {
            let (tokens, errors) = lexer().parse_recovery("(ns foo.bar)");
            assert!(tokens.is_some());
            assert_eq!(errors, vec![]);
            assert_eq!(
                vec![
                    Spanned::new(LexerToken::LPar, 0..1),
                    Spanned::new(
                        LexerToken::Symbol(SymbolToken::Name(String::from("ns"))),
                        1..3,
                    ),
                    Spanned::new(
                        LexerToken::Symbol(SymbolToken::Name(String::from("foo.bar"))),
                        4..11,
                    ),
                    Spanned::new(LexerToken::RPar, 11..12),
                ],
                tokens.unwrap()
            );
        }
    }
    mod literal_expr_parsing {
        use super::*;

        macro_rules! can_parse {
            ($tc_name:ident, $input:expr, $expected:expr) => {
                #[test]
                fn $tc_name() {
                    let (t, errors) = literal_expr_parser().parse_recovery($input);
                    assert!(t.is_some());
                    let (expr, _span) = t.unwrap();
                    assert_eq!($expected, expr);
                }
            };
        }

        can_parse!(nil, [LexerToken::Nil], LiteralExpr::Nil);
        can_parse!(
            boolean_true,
            [LexerToken::Boolean(true)],
            LiteralExpr::Bool(true)
        );
        can_parse!(
            boolean_false,
            [LexerToken::Boolean(false)],
            LiteralExpr::Bool(false)
        );
        can_parse!(
            long,
            [LexerToken::Long(42)],
            LiteralExpr::Number(String::from("42"))
        );
        can_parse!(
            char_simple,
            [LexerToken::Character('x')],
            LiteralExpr::Character('x')
        );
        can_parse!(
            string_simple,
            [LexerToken::String(String::from("foo"))],
            LiteralExpr::Str(String::from("foo"))
        );
        can_parse!(
            symbol_simple,
            [LexerToken::Symbol(SymbolToken::Name(String::from("foo")))],
            LiteralExpr::Symbol(String::from("foo"))
        );
        can_parse!(
            symbol_with_ns,
            [LexerToken::NsSymbol(
                NameToken(String::from("a.b")),
                SymbolToken::Name(String::from("foo"))
            )],
            LiteralExpr::Symbol(String::from("a.b/foo"))
        );
        can_parse!(
            keyword_simple,
            [
                LexerToken::Colon,
                LexerToken::Symbol(SymbolToken::Name(String::from("foo")))
            ],
            LiteralExpr::Keyword(KeywordLiteral::SimpleKeyword(SymbolLiteral::SimpleSymbol(
                SymbolToken::Name(String::from("foo"))
            )))
        );
        can_parse!(
            keyword_macro_keyword,
            [
                LexerToken::Colon,
                LexerToken::Colon,
                LexerToken::Symbol(SymbolToken::Name(String::from("foo")))
            ],
            LiteralExpr::Keyword(KeywordLiteral::MacroKeyword(SymbolLiteral::SimpleSymbol(
                SymbolToken::Name(String::from("foo"))
            )))
        );
    }

    mod form_expr_parsing {
        use super::*;

        macro_rules! can_parse {
            ($tc_name:ident, $input:expr, $matcher:expr) => {
                #[test]
                fn $tc_name() {
                    let (t, errors) = form_expr_parser().parse_recovery($input);
                    assert_eq!(errors, vec![]);
                    assert!(t.is_some());
                    let (expr, span): Spanned<FormExpr> = t.unwrap();
                    println!("{:?}", expr.clone());
                    let is_match = $matcher(expr);
                    assert!(is_match);
                }
            };
        }

        can_parse!(literal_single, [LexerToken::Nil,], |t| match t {
            FormExpr::Literal(le) => match *le {
                (LiteralExpr::Nil, _span) => true,
                _ => false,
            },
            _ => false,
        });

        can_parse!(
            list_flat_single_literal,
            [
                LexerToken::LPar,
                LexerToken::String(String::from("foo")),
                LexerToken::RPar,
            ],
            |t| match t {
                FormExpr::List(le) => match *le {
                    (ListExpr((FormsExpr(forms), _)), _span) => match &(*forms)[..] {
                        [(FormExpr::Literal(lit), _)] => match &**lit {
                            (LiteralExpr::Str(s), _span) => s == "foo",
                            _ => false,
                        },
                        _ => false,
                    },
                    _ => false,
                },
                _ => false,
            }
        );

        can_parse!(
            list_flat_multiple_ns_definition_minimal,
            [
                LexerToken::LPar,
                LexerToken::Symbol(SymbolToken::Name(String::from("ns"))),
                LexerToken::Symbol(SymbolToken::Name(String::from("foo.bar"))),
                LexerToken::RPar,
            ],
            |t| match t {
                FormExpr::List(le) => match *le {
                    (ListExpr((FormsExpr(forms), _)), _span) => match &(*forms)[..] {
                        [(FormExpr::Literal(ns_sym), _), (FormExpr::Literal(name_sym), _)] =>
                            match (&**ns_sym, &**name_sym) {
                                ((LiteralExpr::Symbol(ns), _), (LiteralExpr::Symbol(name), _)) =>
                                    ns == "ns" && name == "foo.bar",
                                _ => false,
                            },
                        _ => false,
                    },
                    _ => false,
                },
                _ => false,
            }
        );
    }
}
