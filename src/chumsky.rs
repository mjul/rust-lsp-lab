//! This is the parser and interpreter for a simplified Clojure grammar
//! See the complete ANTLR 4 grammar here:
//! https://github.com/antlr/grammars-v4/blob/master/clojure/Clojure.g4

use core::fmt;
use std::collections::HashMap;
use std::fmt::{Display, Formatter};

use crate::semantic_token::LEGEND_TYPE;
use chumsky::Parser;
use chumsky::{prelude::*, stream::Stream};
use serde::{Deserialize, Serialize};
use tower_lsp::lsp_types::SemanticTokenType;

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

impl From<&str> for NameToken {
    fn from(value: &str) -> Self {
        NameToken(String::from(value))
    }
}

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
    /// Colon, `:`
    Colon,
    /// Left parenthesis, '('
    LPar,
    /// Right parenthesis, ')'
    RPar,
    /// Left square bracket, '['
    LBra,
    /// Right square bracket, ']'
    RBra,
    /// Left curly bracket, '{'
    LCurl,
    /// Right curly bracket, '}'
    RCurl,
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
            LexerToken::LBra => write!(f, "["),
            LexerToken::RBra => write!(f, "]"),
            LexerToken::LCurl => write!(f, "{}", "{"),
            LexerToken::RCurl => write!(f, "{}", "}"),
        }
    }
}

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
        .then(take_until(
            text::newline::<Simple<char>>().or(end().rewind()),
        ))
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

    let lpar = just::<_, _, Simple<char>>('(').to(LexerToken::LPar);
    let rpar = just::<_, _, Simple<char>>(')').to(LexerToken::RPar);
    let lbra = just::<_, _, Simple<char>>('[').to(LexerToken::LBra);
    let rbra = just::<_, _, Simple<char>>(']').to(LexerToken::RBra);
    let lcurl = just::<_, _, Simple<char>>('{').to(LexerToken::LCurl);
    let rcurl = just::<_, _, Simple<char>>('}').to(LexerToken::RCurl);
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
        lpar, rpar, lbra, rbra, lcurl, rcurl, colon, char_, str_, boolean, nil, number, ns_symbol,
        symbol,
    ))
    .recover_with(skip_then_retry_until([]));

    token
        .map_with_span(|tok, span| (tok, span))
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
    Vector(Box<Spanned<VectorExpr>>),
    Map(Box<Spanned<MapExpr>>),
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
pub struct MapExpr(Vec<(Spanned<FormExpr>, Spanned<FormExpr>)>);

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
    pub body: Spanned<FormsExpr>,
    pub name: Spanned<String>,
    pub span: Span,
}

/// Parse lexer tokens to a literal expression, `LiteralExpr`
fn literal_expr_parser() -> impl Parser<LexerToken, Spanned<LiteralExpr>, Error = Simple<LexerToken>>
{
    let keyword = just(LexerToken::Colon)
        .repeated()
        .at_least(1)
        .at_most(2)
        .then(filter(|t| match t {
            LexerToken::Symbol(_) => true,
            LexerToken::NsSymbol(_, _) => true,
            _ => false,
        }))
        .map_with_span(|(colons, symbol), span| {
            Spanned::new(
                match (colons.len(), symbol) {
                    (1, LexerToken::Symbol(s)) => LiteralExpr::Keyword(
                        KeywordLiteral::SimpleKeyword(SymbolLiteral::SimpleSymbol(s)),
                    ),
                    (1, LexerToken::NsSymbol(n, s)) => LiteralExpr::Keyword(
                        KeywordLiteral::SimpleKeyword(SymbolLiteral::NsSymbol(n, s)),
                    ),
                    (2, LexerToken::Symbol(s)) => LiteralExpr::Keyword(
                        KeywordLiteral::MacroKeyword(SymbolLiteral::SimpleSymbol(s)),
                    ),
                    (2, LexerToken::NsSymbol(n, s)) => LiteralExpr::Keyword(
                        KeywordLiteral::MacroKeyword(SymbolLiteral::NsSymbol(n, s)),
                    ),
                    _ => unreachable!(),
                },
                span,
            )
        });

    let simple_literal = none_of::<LexerToken, _, _>(vec![
        LexerToken::Colon,
        LexerToken::LPar,
        LexerToken::RPar,
        LexerToken::LBra,
        LexerToken::RBra,
        LexerToken::LCurl,
        LexerToken::RCurl,
    ])
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
        LexerToken::LBra => unreachable!(),
        LexerToken::RBra => unreachable!(),
        LexerToken::LCurl => unreachable!(),
        LexerToken::RCurl => unreachable!(),
    });

    choice((keyword, simple_literal))
}

/// Parse lexer tokens to a form expression, `FormExpr`
fn form_expr_parser() -> impl Parser<LexerToken, Spanned<FormExpr>, Error = Simple<LexerToken>> {
    recursive(|form_expr_parser| {
        let forms = form_expr_parser
            .clone()
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
            .clone()
            .delimited_by::<_, _, _, _>(just(LexerToken::LPar), just(LexerToken::RPar))
            .map_with_span(|forms_expr, span: Span| {
                // TODO: fix the span to include the parens
                Spanned::new(
                    FormExpr::List(Box::new(Spanned::new(ListExpr(forms_expr), span.clone()))),
                    span.clone(),
                )
            })
            .labelled("list");

        let vector = forms
            .delimited_by::<_, _, _, _>(just(LexerToken::LBra), just(LexerToken::RBra))
            .map_with_span(|forms_expr, span: Span| {
                // TODO: fix the span to include the brackets
                Spanned::new(
                    FormExpr::Vector(Box::new(Spanned::new(VectorExpr(forms_expr), span.clone()))),
                    span.clone(),
                )
            })
            .labelled("vector");

        let map = form_expr_parser
            .clone()
            .labelled("key")
            .then(form_expr_parser.clone().labelled("value"))
            .repeated()
            .delimited_by::<_, _, _, _>(just(LexerToken::LCurl), just(LexerToken::RCurl))
            .map_with_span::<_, _>(|kw_pairs, span: Span| {
                // TODO: fix the span to include the curly brackets
                Spanned::new(
                    FormExpr::Map(Box::new(Spanned::new(MapExpr(kw_pairs), span.clone()))),
                    span.clone(),
                )
            })
            .labelled("map");

        // TODO: add set etc.
        choice((literal, list, vector, map)).labelled("form")
    })
}

/// Parse a lexer token stream of a source file to a `FileExpr`.
fn file_expr_parser() -> impl Parser<LexerToken, FileExpr, Error = Simple<LexerToken>> {
    form_expr_parser()
        .repeated()
        .then_ignore(end())
        // No span since this is the whole file anyway
        .map(|forms| FileExpr::Forms(forms))
}

#[deprecated]
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

/// A `defn` declaration
struct Defn {
    defn: Spanned<String>,
    name: Spanned<String>,
    doc_comment: Option<Spanned<String>>,
    args: Spanned<String>,
    body: Spanned<FormsExpr>,
    span: Span,
}

/// Parse a `FormExpr` into a `Defn` if it matches the proper `(defn f [] ...)` list form.
pub fn defn_parser() -> impl Parser<FormExpr, Defn, Error = Simple<FormExpr>> + Clone {
    filter_map::<_, Defn, _, Simple<FormExpr>>(|form_span, form| {
        match &form {
            FormExpr::List(bsle) => {
                match &**bsle {
                    (ListExpr((list_forms, _list_forms_span)), list_expr_span) => {
                        match list_forms {
                            FormsExpr(vec_form_expr) => match &vec_form_expr[..] {
                                [(FormExpr::Literal(defn_form_expr), defn_span), (FormExpr::Literal(name_form_expr), name_span), rest @ ..] =>
                                {
                                    match (&**defn_form_expr, &**name_form_expr) {
                                        // Starts with two symbols, e.g. (defn foo [x] (inc x)
                                        (
                                            (LiteralExpr::Symbol(defn), defn_span),
                                            (LiteralExpr::Symbol(name), name_span),
                                        ) => {
                                            if "defn" == defn {
                                                // everything in the list except the two initial symbols for defn and the name
                                                let rest_forms: Vec<Spanned<FormExpr>> =
                                                    rest.into_iter().map(|x| x.clone()).collect();
                                                let rest_forms_span = if rest_forms.is_empty() {
                                                    _list_forms_span.end.._list_forms_span.end
                                                } else {
                                                    let start =
                                                        rest_forms.first().unwrap().1.start.clone();
                                                    let end = rest_forms.last().unwrap().1.end;
                                                    start..end
                                                };
                                                // TODO: elaborate this
                                                Ok(Defn {
                                                    defn: (defn.clone(), defn_span.clone()),
                                                    name: (name.clone(), name_span.clone()),
                                                    doc_comment: None,
                                                    args: ("".to_string(), Default::default()),
                                                    body: Spanned::new(
                                                        FormsExpr(Box::new(rest_forms)),
                                                        rest_forms_span,
                                                    ),
                                                    span: list_expr_span.clone(),
                                                })
                                            } else {
                                                Err(Simple::custom(
                                                    defn_span.clone(),
                                                    "Initial symbol is not defn",
                                                ))
                                            }
                                        }
                                        _ => Err(Simple::custom(
                                            list_expr_span.clone(),
                                            "List does not start with two symbols",
                                        )),
                                    }
                                }
                                _ => Err(Simple::custom(
                                    list_expr_span.clone(),
                                    "List does not start with two literals",
                                )),
                            },
                        }
                    }
                }
            }
            _ => Err(Simple::custom(form_span.clone(), "Form is not a list")),
        }
    })
}

impl TryFrom<Spanned<ListExpr>> for Defn {
    type Error = ();
    fn try_from(value: Spanned<ListExpr>) -> Result<Self, Self::Error> {
        let defn = defn_parser().parse(vec![FormExpr::List(Box::new(value))]);
        defn.map_err(|e| ())
    }
}

/// Parser that extracts the functions
pub fn funcs_parser(
) -> impl Parser<FileExpr, HashMap<String, Func>, Error = Simple<FileExpr>> + Clone {
    // TODO: simplify using the defn_parser above

    fn try_get_list_symbol_symbol_rest(
        (form, form_span): &Spanned<FormExpr>,
    ) -> Option<((String, Span), (String, Span), Vec<Spanned<FormExpr>>, Span)> {
        match &form {
            FormExpr::Literal(_) => None,
            FormExpr::List(sle) => match &**sle {
                (ListExpr((list_forms, _list_forms_span)), _list_expr_span) => match list_forms {
                    FormsExpr(vec_form_expr) => match &vec_form_expr[..] {
                        [(FormExpr::Literal(defn_form_expr), defn_span), (FormExpr::Literal(name_form_expr), name_span), rest @ ..] =>
                        {
                            match (&**defn_form_expr, &**name_form_expr) {
                                // Starts with two symbols, e.g. (defn foo [x] (inc x)
                                (
                                    (LiteralExpr::Symbol(defn), _),
                                    (LiteralExpr::Symbol(name), _),
                                ) => {
                                    // everything in the list except the two initial symbols for defn and the name
                                    let rest_forms = rest.into_iter().map(|x| x.clone()).collect();
                                    Some((
                                        (defn.clone(), defn_span.clone()),
                                        (name.clone(), name_span.clone()),
                                        rest_forms,
                                        form_span.clone(),
                                    ))
                                }
                                _ => None,
                            }
                        }
                        _ => None,
                    },
                },
            },
            FormExpr::Vector(_) => None,
            FormExpr::Map(_) => None,
        }
    }

    any::<_, Simple<FileExpr>>().then_ignore(end()).map(|x| {
        let FileExpr::Forms(top_level_forms) = x;
        let lists_with_two_symbol_heads = top_level_forms
            .iter()
            .filter_map(|x| try_get_list_symbol_symbol_rest(x));
        let defns = lists_with_two_symbol_heads
            .filter(|((s1, s1_span), (name, name_span), _rest_forms, _form_span)| s1 == "defn");
        let mut funcs = HashMap::<String, Func>::new();
        for ((s1, s1_span), (name, name_span), rest_forms, form_span) in defns {
            let mut inner = rest_forms.into_iter().peekable();
            match inner.peek() {
                Some((FormExpr::Literal(le), _span)) => match &**le {
                    (LiteralExpr::Str(_doc_comment), _span) => {
                        inner.next();
                    }
                    _ => {}
                },
                _ => {}
            }
            match inner.peek() {
                Some((FormExpr::Vector(_args), _span)) => {
                    inner.next();
                }
                _ => {}
            }
            let body_forms: Vec<Spanned<FormExpr>> = inner.collect();
            let body_span_start = body_forms.first().map_or(form_span.end, |x| x.1.start);
            let body_span_end = body_forms.last().map_or(form_span.end, |x| x.1.end);
            let body_span = body_span_start..body_span_end;
            let func = Func {
                args: vec![],
                body: (FormsExpr(Box::new(body_forms)), body_span),
                name: (name.to_string(), name_span.clone()),
                span: form_span.clone(),
            };
            funcs.insert(name.clone(), func);
        }
        funcs
    })
}

/// Try to infer the type of a `FormsExpr`.
/// Adds the spans and their types to the `symbol_type_table`.
pub fn type_inference(
    expr: &Spanned<FormsExpr>,
    symbol_type_table: &mut HashMap<Span, LiteralExpr>,
) {
    match &expr.0 {
        FormsExpr(bfs) => bfs
            .iter()
            .for_each(|fe| type_inference_form_expr(fe, symbol_type_table)),
    }
}

/// Try to infer the type of a `FormExpr`.
/// Adds the spans and their types to the `symbol_type_table`.
fn type_inference_form_expr(
    expr: &Spanned<FormExpr>,
    symbol_type_table: &mut HashMap<Span, LiteralExpr>,
) {
    match &expr.0 {
        FormExpr::Literal(ble) => {}
        FormExpr::List(le) => {
            // If it is a defn register a type under that symbol name
            match Defn::try_from(*le.clone()) {
                Ok(defn) => {
                    // TODO: fill out the type
                    symbol_type_table.insert(defn.name.1.clone(), LiteralExpr::Nil);
                }
                _ => {}
            }
            // TODO: implement for def and let bindings
        }
        FormExpr::Vector(_) => {}
        FormExpr::Map(_) => {}
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
    let (tokens, lexer_errors) = lexer().parse_recovery(src);

    //log::debug!("Lexer Tokens: {:?}", tokens);
    //log::debug!("Lexer Errors: {:?}", lexer_errors);

    let (ast, tokenize_errors, semantic_tokens) = if let Some(tokens) = tokens {
        // First we collect the semantic tokens for syntax highlighting from the lexer tokens
        // info!("Tokens = {:?}", tokens);
        let semantic_tokens = tokens
            .iter()
            .filter_map(|(token, span)| {
                let token_type = match token {
                    LexerToken::Colon
                    | LexerToken::LPar
                    | LexerToken::RPar
                    | LexerToken::LBra
                    | LexerToken::RBra
                    | LexerToken::LCurl
                    | LexerToken::RCurl => None,
                    LexerToken::Nil | LexerToken::Boolean(_) => {
                        Some(SemanticTokenType::KEYWORD.clone())
                    }
                    LexerToken::Character(_) | LexerToken::String(_) => {
                        Some(SemanticTokenType::STRING.clone())
                    }
                    LexerToken::Long(_) => Some(SemanticTokenType::NUMBER.clone()),
                    LexerToken::Symbol(_) | LexerToken::NsSymbol(_, _) => {
                        Some(SemanticTokenType::VARIABLE.clone())
                    }
                };
                token_type.map(|tt| ImCompleteSemanticToken {
                    start: span.start,
                    length: span.len(),
                    token_type: LEGEND_TYPE.iter().position(|item| item == &tt).unwrap(),
                })
            })
            .collect::<Vec<_>>();

        let len = src.chars().count();
        // Now parse the lexemes (tokens) into an AST
        let (file_expr, file_parse_errs) =
            file_expr_parser().parse_recovery(Stream::from_iter(len..len + 1, tokens.into_iter()));

        // Then, extract the function definitions
        let (functions_by_name, parse_errs) =
            funcs_parser().parse_recovery(file_expr.map_or(vec![], |fe| vec![fe]));

        (functions_by_name, file_parse_errs, semantic_tokens)
    } else {
        (None, Vec::new(), vec![])
    };

    let parse_errors = lexer_errors
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
        can_parse!(parenthesis_left, "(", LexerToken::LPar);
        can_parse!(parenthesis_right, ")", LexerToken::RPar);
        can_parse!(square_bracket_left, "[", LexerToken::LBra);
        can_parse!(square_bracket_right, "]", LexerToken::RBra);
        can_parse!(curly_bracket_left, "{", LexerToken::LCurl);
        can_parse!(curly_bracket_right, "}", LexerToken::RCurl);
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
        can_parse!(comment_at_beginning, ";; Comment\n1", LexerToken::Long(1));
        can_parse!(
            comment_at_end_no_newline,
            "1\n;; Comment",
            LexerToken::Long(1)
        );
        can_parse!(
            comment_at_end_with_newline,
            "1\n;; Comment\n",
            LexerToken::Long(1)
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
            :
            ()
            []
            {}
            "#,
            );
            assert_eq!(true, actual.is_ok());
            let tokens = actual.unwrap();
            assert_eq!(16, tokens.len());
            let lts: Vec<LexerToken> = tokens.into_iter().map(|(t, s)| t).collect();
            assert_eq!(LexerToken::Nil, lts[0]);
            assert_eq!(LexerToken::Boolean(true), lts[1]);
            assert_eq!(LexerToken::Boolean(false), lts[2]);
            assert_eq!(LexerToken::Long(1), lts[3]);
            assert_eq!(LexerToken::String("foo-string".to_string()), lts[4]);
            assert_eq!(LexerToken::Symbol(SymbolToken::Dot), lts[5]);
            assert_eq!(LexerToken::Symbol(SymbolToken::Slash), lts[6]);
            assert_eq!(
                LexerToken::Symbol(SymbolToken::Name(String::from("foo"))),
                lts[7]
            );
            assert_eq!(
                LexerToken::NsSymbol(
                    NameToken(String::from("foo.bar")),
                    SymbolToken::Name(String::from("baz")),
                ),
                lts[8]
            );
            assert_eq!(LexerToken::Colon, lts[9]);
            assert_eq!(LexerToken::LPar, lts[10]);
            assert_eq!(LexerToken::RPar, lts[11]);
            assert_eq!(LexerToken::LBra, lts[12]);
            assert_eq!(LexerToken::RBra, lts[13]);
            assert_eq!(LexerToken::LCurl, lts[14]);
            assert_eq!(LexerToken::RCurl, lts[15]);
        }

        #[test]
        fn lexer_can_parse_keyword_variants() {
            let source = r#"
            :keyword
            :kw/keyword
            :k.w/keyword
            ::keyword
            ::kw/keyword
            ::k.w/keyword
            "#;
            let (tokens, errors) = lexer().parse_recovery(source);
            assert_eq!(errors, vec![]);
            let tokens: Vec<LexerToken> = tokens.unwrap().into_iter().map(|(t, s)| t).collect();
            assert_eq!(
                vec![
                    LexerToken::Colon,
                    LexerToken::Symbol(SymbolToken::Name(String::from("keyword"))),
                    LexerToken::Colon,
                    LexerToken::NsSymbol(
                        NameToken::from("kw"),
                        SymbolToken::Name(String::from("keyword"))
                    ),
                    LexerToken::Colon,
                    LexerToken::NsSymbol(
                        NameToken::from("k.w"),
                        SymbolToken::Name(String::from("keyword"))
                    ),
                    LexerToken::Colon,
                    LexerToken::Colon,
                    LexerToken::Symbol(SymbolToken::Name(String::from("keyword"))),
                    LexerToken::Colon,
                    LexerToken::Colon,
                    LexerToken::NsSymbol(
                        NameToken::from("kw"),
                        SymbolToken::Name(String::from("keyword"))
                    ),
                    LexerToken::Colon,
                    LexerToken::Colon,
                    LexerToken::NsSymbol(
                        NameToken::from("k.w"),
                        SymbolToken::Name(String::from("keyword"))
                    ),
                ],
                tokens
            );
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
                    let empty_errors: Vec<Simple<LexerToken>> = vec![];
                    assert_eq!(empty_errors, errors);
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
            keyword_simple_with_simple_symbol,
            [
                LexerToken::Colon,
                LexerToken::Symbol(SymbolToken::Name(String::from("foo")))
            ],
            LiteralExpr::Keyword(KeywordLiteral::SimpleKeyword(SymbolLiteral::SimpleSymbol(
                SymbolToken::Name(String::from("foo"))
            )))
        );
        can_parse!(
            keyword_simple_with_ns_symbol,
            [
                LexerToken::Colon,
                LexerToken::NsSymbol(
                    NameToken(String::from("foo.bar")),
                    SymbolToken::Name(String::from("baz"))
                )
            ],
            LiteralExpr::Keyword(KeywordLiteral::SimpleKeyword(SymbolLiteral::NsSymbol(
                NameToken(String::from("foo.bar")),
                SymbolToken::Name(String::from("baz"))
            )))
        );
        can_parse!(
            keyword_macro_keyword_with_simple_symbol,
            [
                LexerToken::Colon,
                LexerToken::Colon,
                LexerToken::Symbol(SymbolToken::Name(String::from("foo")))
            ],
            LiteralExpr::Keyword(KeywordLiteral::MacroKeyword(SymbolLiteral::SimpleSymbol(
                SymbolToken::Name(String::from("foo"))
            )))
        );
        can_parse!(
            keyword_macro_keyword_with_ns_symbol,
            [
                LexerToken::Colon,
                LexerToken::Colon,
                LexerToken::NsSymbol(
                    NameToken(String::from("foo.bar")),
                    SymbolToken::Name(String::from("baz"))
                ),
            ],
            LiteralExpr::Keyword(KeywordLiteral::MacroKeyword(SymbolLiteral::NsSymbol(
                NameToken(String::from("foo.bar")),
                SymbolToken::Name(String::from("baz"))
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

        can_parse!(
            vector_flat_empty,
            [LexerToken::LBra, LexerToken::RBra,],
            |t| match t {
                FormExpr::Vector(ve) => match *ve {
                    (VectorExpr((FormsExpr(forms), _)), _span) => forms.is_empty(),
                    _ => false,
                },
                _ => false,
            }
        );

        can_parse!(
            vector_flat_single_literal,
            [LexerToken::LBra, LexerToken::Long(1), LexerToken::RBra,],
            |t| match t {
                FormExpr::Vector(ve) => match *ve {
                    (VectorExpr((FormsExpr(forms), _)), _span) => match &(*forms)[..] {
                        [(FormExpr::Literal(lit), _)] => match &**lit {
                            (LiteralExpr::Number(s), _span) => s == "1",
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
            map_flat_empty,
            [LexerToken::LCurl, LexerToken::RCurl,],
            |t| match t {
                FormExpr::Map(expr) => match *expr {
                    (MapExpr(key_val_pairs), _span) => key_val_pairs.is_empty(),
                    _ => false,
                },
                _ => false,
            }
        );

        can_parse!(
            map_flat_single_key_val,
            [
                LexerToken::LCurl,
                LexerToken::Colon,
                LexerToken::Symbol(SymbolToken::Name(String::from("number"))),
                LexerToken::Long(42),
                LexerToken::RCurl,
            ],
            |t| match t {
                FormExpr::Map(me) => match *me {
                    (MapExpr(pairs), _) => match &(*pairs)[..] {
                        [((FormExpr::Literal(key), _), (FormExpr::Literal(val), _))] =>
                            match &(*key.clone(), *val.clone()) {
                                (
                                    (
                                        LiteralExpr::Keyword(KeywordLiteral::SimpleKeyword(
                                            SymbolLiteral::SimpleSymbol(SymbolToken::Name(key)),
                                        )),
                                        _,
                                    ),
                                    (LiteralExpr::Number(num), _),
                                ) => key == "number" && num == "42",
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
            map_flat_multiple_key_val,
            [
                LexerToken::LCurl,
                LexerToken::String(String::from("a")),
                LexerToken::Long(1),
                LexerToken::String(String::from("b")),
                LexerToken::Long(2),
                LexerToken::RCurl,
            ],
            |t| match t {
                FormExpr::Map(me) => match *me {
                    (MapExpr(pairs), _) => match &(*pairs)[..] {
                        [((FormExpr::Literal(key1), _), (FormExpr::Literal(val1), _)), ((FormExpr::Literal(key2), _), (FormExpr::Literal(val2), _))] =>
                            match &(*key1.clone(), *val1.clone(), *key2.clone(), *val2.clone()) {
                                (
                                    (LiteralExpr::Str(k1), _),
                                    (LiteralExpr::Number(v1), _),
                                    (LiteralExpr::Str(k2), _),
                                    (LiteralExpr::Number(v2), _),
                                ) => k1 == "a" && v1 == "1" && k2 == "b" && v2 == "2",
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

    mod file_expr_parsing {
        use super::*;
        fn parse(input: Vec<LexerToken>) -> FileExpr {
            let (t, errors) = file_expr_parser().parse_recovery(input);
            assert_eq!(errors, vec![]);
            assert!(t.is_some());
            let expr = t.unwrap();
            expr
        }

        #[test]
        fn can_parse_empty_file() {
            let actual = parse(vec![]);
            assert_eq!(FileExpr::Forms(vec![]), actual);
        }

        #[test]
        fn can_parse_file_with_ns() {
            let actual = parse(vec![
                // (ns foo.bar)
                LexerToken::LPar,
                LexerToken::Symbol(SymbolToken::Name(String::from("ns"))),
                LexerToken::Symbol(SymbolToken::Name(String::from("foo.bar"))),
                LexerToken::RPar,
            ]);
            let FileExpr::Forms(forms) = actual;
            assert_eq!(1, forms.len());
            let (form, span) = &forms[0];
            assert!(form.is_list());
        }

        #[test]
        fn can_parse_file_with_ns_and_code() {
            let actual = parse(vec![
                // (ns foo.bar)
                LexerToken::LPar,
                LexerToken::Symbol(SymbolToken::Name(String::from("ns"))),
                LexerToken::Symbol(SymbolToken::Name(String::from("foo.bar"))),
                LexerToken::RPar,
                // (def n 42)
                LexerToken::LPar,
                LexerToken::Symbol(SymbolToken::Name(String::from("def"))),
                LexerToken::Symbol(SymbolToken::Name(String::from("n"))),
                LexerToken::Long(42),
                LexerToken::RPar,
            ]);
            let FileExpr::Forms(forms) = actual;
            assert_eq!(2, forms.len());
            assert!(forms[0].0.is_list());
            assert!(forms[1].0.is_list());
        }
    }

    mod defn_parsing {
        use super::*;

        #[test]
        fn can_parse_defn_with_no_args_no_comment_single_expr_body() {
            // (defn f [] nil)
            let le = Spanned::new(
                ListExpr((
                    FormsExpr(Box::new(vec![
                        Spanned::new(
                            FormExpr::Literal(Box::new(Spanned::new(
                                LiteralExpr::Symbol(String::from("defn")),
                                1..5,
                            ))),
                            1..5,
                        ),
                        Spanned::new(
                            FormExpr::Literal(Box::new(Spanned::new(
                                LiteralExpr::Symbol(String::from("f")),
                                6..7,
                            ))),
                            6..7,
                        ),
                        Spanned::new(
                            FormExpr::Vector(Box::new(Spanned::new(
                                VectorExpr(Spanned::new(FormsExpr(Box::new(vec![])), 8..10)),
                                8..10,
                            ))),
                            8..10,
                        ),
                        Spanned::new(
                            FormExpr::Literal(Box::new(Spanned::new(LiteralExpr::Nil, 11..14))),
                            11..14,
                        ),
                    ])),
                    1..14,
                )),
                0..15,
            );
            let fe = FormExpr::List(Box::new(le));
            let (actual_defn, actual_errors) = defn_parser().parse_recovery(vec![fe]);
            assert!(actual_errors.is_empty());
            let actual = actual_defn.unwrap();
            assert_eq!((String::from("defn"), 1..5), actual.defn);
            assert_eq!((String::from("f"), 6..7), actual.name);
            // TODO: elaborate
        }
    }

    mod parse_integration {
        use super::*;

        #[test]
        fn can_parse_ns_form() {
            let source = "(ns foo.bar)";

            let actual = parse(source);
            let ParserResult {
                ast,
                parse_errors,
                semantic_tokens,
            } = actual;

            assert_eq!(
                (1, 2),
                (semantic_tokens[0].start, semantic_tokens[0].length)
            ); // ns
            assert_eq!(
                (4, 7),
                (semantic_tokens[1].start, semantic_tokens[1].length)
            ); // foo.bar

            assert_eq!(parse_errors, vec![]);

            assert!(ast.is_some());
            assert!(ast.unwrap().is_empty());
        }

        #[test]
        fn can_parse_ns_form_with_defns() {
            let source = r#"(ns foo.bar)
                   (defn f [x] (inc x))
                   (defn g [x] (f (f x)))
                "#;

            let actual = parse(source);
            let ParserResult {
                ast,
                parse_errors,
                semantic_tokens,
            } = actual;

            assert_eq!(
                (1, 2),
                (semantic_tokens[0].start, semantic_tokens[0].length)
            ); // ns
            assert_eq!(
                (4, 7),
                (semantic_tokens[1].start, semantic_tokens[1].length)
            ); // foo.bar

            assert_eq!(parse_errors, vec![]);

            assert!(ast.is_some());

            let funcs = ast.unwrap();

            // Check defn of f
            let actual_f = funcs.get("f").unwrap();
            let Func {
                args,
                body,
                name,
                span,
            } = actual_f;
            assert_eq!(Spanned::new(String::from("f"), 38..39), *name);
            assert_eq!(32..52, *span);
            assert_eq!("f", &source[name.1.clone()]);
            assert_eq!("(defn f [x] (inc x))", &source[span.clone()]);
            // TODO: test body

            // Check defn of g
            let actual_f = funcs.get("g").unwrap();
            let Func {
                args,
                body,
                name,
                span,
            } = actual_f;
            assert_eq!(Spanned::new(String::from("g"), 78..79), *name);
            assert_eq!(72..94, *span);
            assert_eq!("g", &source[name.1.clone()]);
            assert_eq!("(defn g [x] (f (f x)))", &source[span.clone()]);
            // TODO: test body
        }

        #[test]
        fn can_parse_to_semantic_tokens() {
            let source = "(ns foo.bar) (def kws [:a :a.b/c ::x ::x.y/z])";
            let actual = parse(source);
            assert_eq!(actual.parse_errors, vec![]);
            println!("Semantics: {:?}", actual.semantic_tokens);
            assert!(!actual.semantic_tokens.is_empty());
            // TODO: check semantic tokens
        }
    }
}
