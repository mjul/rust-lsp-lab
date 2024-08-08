use core::fmt;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use chumsky::{prelude::*, stream::Stream};
use chumsky::Parser;
use serde::{Deserialize, Serialize};
use tower_lsp::lsp_types::SemanticTokenType;

use crate::semantic_token::LEGEND_TYPE;

/// This is the parser and interpreter for a simplified Clojure grammar
/// See the complete ANTLR 4 grammar here:
/// https://github.com/antlr/grammars-v4/blob/master/clojure/Clojure.g4

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
}

impl Display for LexerToken {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            LexerToken::String(s) => write!(f, "\"{}\"", s),
            LexerToken::Long(n) => write!(f, "{}", n),
            LexerToken::Nil => write!(f, "nil"),
            LexerToken::Boolean(b) => write!(f, "{}", match b {
                true => "true",
                false => "false"
            }),
            LexerToken::Symbol(s) => write!(f, "{}", s),
            LexerToken::NsSymbol(n, s) => write!(f, "{}/{}", n, s)
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


fn lexer() -> impl Parser<char, Vec<(LexerToken, Span)>, Error=Simple<char>> {
    // A parser for comments
    let comment = just(";").then(take_until(just('\n'))).padded();

    // A parser for simple strings
    let str_ = just('"')
        .ignore_then(filter(|c| *c != '"').repeated())
        .then_ignore(just('"'))
        .collect::<String>()
        .map(LexerToken::String);

    // A parser for longs
    let long_number =
        text::int::<_, Simple<char>>(10)
            .map(|s| LexerToken::Long(s.parse().unwrap()));
    // TODO: negative numbers
    // TODO: floats
    // TODO: floats, other bases
    /*
    let float_number = text::int(10)
        .chain::<char, _, _>(just('.').chain(text::digits(10)).or_not().flatten())
        .collect::<String>()
        .map(|s| LexerToken::Float(Float::parse(s)));
    */
    let number = long_number;

    // A parser for characters (simplified)
    /*
    let char_ = just('\'')
        .ignore_then(filter(|c| *c != '\''))
        .then_ignore(just('\''))
        .collect::<String>()
        .map(|s| Literal::Character('x'));
    */

    // TODO: use whitespace lexer for last part:
    let symbol_head = none_of::<_, _, Simple<char>>("0123456789^`\\\"#~@:/%()[]{} \n\r\t");
    let symbol_rest =
        choice::<_, Simple<char>>((
            symbol_head.clone(),
            one_of("0123456789"),
            just('.')));

    let name =
        symbol_head
            .then(symbol_rest.clone())
            .then(just::<_, _, Simple<char>>(':')
                .then(symbol_rest.repeated().at_least(1)))
            .map(|((sh, rs), (c, rest))| {
                let mut s = String::new();
                s.push(sh);
                s.push(rs);
                s.push(c);
                for x in rest { s.push(x); }
                NameToken(s)
            }
            );

    let symbol =
        name
            .map(|NameToken(nt)| LexerToken::Symbol(SymbolToken::Name(nt)));
    /*
        choice((
            just("."),
            //just("/"),
            name.map(|NameToken(s)| s)))
            .map(SymbolToken);
            */

    /*
    let ns_symbol =
        name
            .then(just('/'))
            .then(symbol)
            .map(|a, b| LexerToken::NsSymbol(a, b));
*/
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

    let boolean_true = text::keyword("true").map(|_| LexerToken::Boolean(true));
    let boolean_false = text::keyword("false").map(|_| LexerToken::Boolean(false));
    let boolean = boolean_true.or(boolean_false);

    let nil = text::keyword("nil").map(|_| LexerToken::Nil);

    // A single token can be one of the above
    let token =
        choice((
            str_,
            boolean,
            nil,
            number,
            symbol))
            .recover_with(skip_then_retry_until([]));

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

#[derive(Debug)]
pub enum FormExpr {
    Literal(LiteralExpr),
    List(ListExpr),
    Vector(VectorExpr),
    Map(MapExpr),
    // No reader macros
    // ReaderMacro(ReaderMacroExpr)
}

#[derive(Debug)]
pub enum LiteralExpr {}

#[derive(Debug)]
pub enum ListExpr {}

#[derive(Debug)]
pub enum VectorExpr {}

#[derive(Debug)]
pub enum MapExpr {}

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

pub fn funcs_parser() -> impl Parser<LexerToken, HashMap<String, Func>, Error=Simple<LexerToken>> + Clone {
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
                LexerToken::Nil | LexerToken::Boolean(_) => Some(ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE
                        .iter()
                        .position(|item| item == &SemanticTokenType::KEYWORD)
                        .unwrap(),
                }),
                LexerToken::String(_) => Some(ImCompleteSemanticToken {
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
                LexerToken::Symbol(_) => { todo!() }
                LexerToken::NsSymbol(_, _) => { todo!() }
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


#[cfg(test)]
mod tests {
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

    #[test]
    fn lexer_can_parse_nil() {
        let (t, _span) = lex_single("nil");
        assert_eq!(LexerToken::Nil, t);
    }

    #[test]
    fn lexer_can_parse_boolean_true() {
        let (t, _span) = lex_single("true");
        assert_eq!(LexerToken::Boolean(true), t);
    }
    #[test]
    fn lexer_can_parse_boolean_false() {
        let (t, _span) = lex_single("false");
        assert_eq!(LexerToken::Boolean(false), t);
    }

    #[test]
    fn lexer_can_parse_string() {
        let (t, _span) = lex_single("\"foo\"");
        assert_eq!(LexerToken::String(String::from("foo")), t);
    }

    #[test]
    fn lexer_can_parse_symbol_dot() {
        let (t, _span) = lex_single(".");
        assert_eq!(LexerToken::Symbol(SymbolToken::Dot), t);
    }

    #[test]
    fn lexer_can_parse_symbol_slash() {
        let (t, span) = lex_single("/");
        assert_eq!(LexerToken::Symbol(SymbolToken::Slash), t);
    }

    #[test]
    fn lexer_can_parse_symbol_name_simple() {
        let (t, _span) = lex_single("foo");
        assert_eq!(LexerToken::Symbol(SymbolToken::Name(String::from("foo"))), t);
    }

    #[test]
    fn lexer_can_parse_symbol_name_dashed() {
        let (t, _span) = lex_single("foo-bar");
        assert_eq!(LexerToken::Symbol(SymbolToken::Name(String::from("foo-bar"))), t);
    }

    #[test]
    fn lexer_can_parse_symbol_with_namespace() {
        let (t, _span) = lex_single("name.space/foo");
        assert_eq!(LexerToken::NsSymbol(NameToken(String::from("name.space")), SymbolToken::Name(String::from("foo"))), t);
    }
}