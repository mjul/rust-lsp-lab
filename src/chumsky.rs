use core::fmt;
use std::collections::HashMap;
use std::fmt::Formatter;

use chumsky::{prelude::*, stream::Stream};
use chumsky::Parser;
use serde::{Deserialize, Serialize};
use tower_lsp::lsp_types::SemanticTokenType;

use crate::semantic_token::LEGEND_TYPE;

/// This is the parser and interpreter for the 'Foo' language. See `tutorial.md` in the repository's root to learn
/// about it.
pub type Span = std::ops::Range<usize>;
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
pub struct SymbolToken(String);

impl std::fmt::Display for SymbolToken {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}


#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum SymbolLiteral {
    NsSymbol(NameToken, SymbolToken),
    SimpleSymbol(SymbolToken),
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
pub enum Literal {
    Nil,
    Str(String),
    Number(String),
    Character(char),
    Bool(bool),
    Keyword(KeywordLiteral),
    Symbol(String),
}


impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Literal::Nil => write!(f, "nil"),
            Literal::Bool(x) => write!(f, "{}", x),
            Literal::Number(n) => write!(f, "{}", n),
            Literal::Str(s) => write!(f, "{}", s),
            Literal::Character(c) => write!(f, "{}", c),
            Literal::Keyword(kwt) => write!(f, "{}", kwt),
            Literal::Symbol(st) => write!(f, "{}", st),
        }
    }
}


impl fmt::Display for SymbolLiteral {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            SymbolLiteral::NsSymbol(ns, s) => write!(f, ":{}/{}", ns, s),
            SymbolLiteral::SimpleSymbol(s) => write!(f, ":{}", s),
        }
    }
}


fn lexer() -> impl Parser<char, Vec<(Literal, Span)>, Error=Simple<char>> {
    // A parser for numbers
    let num = text::int(10)
        .chain::<char, _, _>(just('.').chain(text::digits(10)).or_not().flatten())
        .collect::<String>()
        .map(Literal::Number);

    // A parser for characters (simplified)
    /*
    let char_ = just('\'')
        .ignore_then(filter(|c| *c != '\''))
        .then_ignore(just('\''))
        .collect::<String>()
        .map(|s| Literal::Character('x'));
    */

    // A parser for strings
    let str_ = just('"')
        .ignore_then(filter(|c| *c != '"').repeated())
        .then_ignore(just('"'))
        .collect::<String>()
        .map(Literal::Number);
    /*
        let symbol_ = just(':')
            .ignore_then(symbol_)
            .collect::<String>()
            .map(KeywordLiteral::SimpleKeyword);

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

    // A parser for identifiers and keywords
    let ident = text::ident().map(|ident: String| match ident.as_str() {
        "true" => Literal::Bool(true),
        "false" => Literal::Bool(false),
        "nil" => Literal::Nil,
        s => Literal::Symbol(s.to_string()),
    });

    // A single token can be one of the above
    let token = num
        //.or(char_)
        .or(str_)
        .or(ident)
        .recover_with(skip_then_retry_until([]));

    let comment = just(";").then(take_until(just('\n'))).padded();

    token
        .padded_by(comment.repeated())
        .map_with_span(|tok, span| (tok, span))
        .padded()
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

// An expression node in the AST. Children are spanned so we can generate useful runtime errors.
#[derive(Debug)]
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
#[derive(Debug)]
pub struct Func {
    pub args: Vec<Spanned<String>>,
    pub body: Spanned<Expr>,
    pub name: Spanned<String>,
    pub span: Span,
}

fn expr_parser() -> impl Parser<Literal, Spanned<Expr>, Error=Simple<Literal>> + Clone {
    recursive(|expr| {
        let raw_expr = recursive(|raw_expr| {
            let val = filter_map(|span, tok| match tok {
                Literal::Nil => Ok(Expr::Value(Value::Null)),
                Literal::Bool(x) => Ok(Expr::Value(Value::Bool(x))),
                Literal::Number(n) => Ok(Expr::Value(Value::Num(n.parse().unwrap()))),
                Literal::Str(s) => Ok(Expr::Value(Value::Str(s))),
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

pub fn funcs_parser() -> impl Parser<Literal, HashMap<String, Func>, Error=Simple<Literal>> + Clone {
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

pub fn parse(src: &str) -> ParserResult {
    let (tokens, errs) = lexer().parse_recovery(src);

    let (ast, tokenize_errors, semantic_tokens) = if let Some(tokens) = tokens {
        // info!("Tokens = {:?}", tokens);
        let semantic_tokens = tokens
            .iter()
            .filter_map(|(token, span)| match token {
                Literal::Nil => None,
                Literal::Bool(_) => None,

                Literal::Number(_) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::NUMBER)
                        .unwrap(),
                }),
                Literal::Str(_) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::STRING)
                        .unwrap(),
                }),
                Literal::Character(_) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::STRING)
                        .unwrap(),
                }),
                Literal::Keyword(kwl) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::STRING)
                        .unwrap(),
                }),
                Literal::Symbol(kwl) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::FUNCTION)
                        .unwrap(),
                }),
            })
            .collect::<Vec<_>>();
        let len = src.chars().count();
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
