use std::collections::HashMap;

use crate::chumsky::{FormsExpr, Func, Spanned};

#[derive(Debug, PartialEq)]
pub enum ImCompleteCompletionItem {
    Variable(String),
    Function(String, Vec<String>),
}
/// return (need_to_continue_search, founded reference)
pub fn completion(
    ast: &HashMap<String, Func>,
    ident_offset: usize,
) -> HashMap<String, ImCompleteCompletionItem> {
    let mut map = HashMap::new();
    for (_, v) in ast.iter() {
        if v.name.1.end < ident_offset {
            map.insert(
                v.name.0.clone(),
                ImCompleteCompletionItem::Function(
                    v.name.0.clone(),
                    v.args.clone().into_iter().map(|(name, _)| name).collect(),
                ),
            );
        }
    }

    // collect params variable
    for (_, v) in ast.iter() {
        if v.span.end > ident_offset && v.span.start < ident_offset {
            // log::debug!("this is completion from body {}", name);
            v.args.iter().for_each(|(item, _)| {
                map.insert(
                    item.clone(),
                    ImCompleteCompletionItem::Variable(item.clone()),
                );
            });
            get_completion_of(&v.body, &mut map, ident_offset);
        }
    }
    map
}

pub fn get_completion_of(
    expr: &Spanned<FormsExpr>,
    definition_map: &mut HashMap<String, ImCompleteCompletionItem>,
    ident_offset: usize,
) -> bool {
    /*
    match &expr.0 {
        Expr::Error => true,
        Expr::Value(_) => true,
        Expr::List(exprs) => exprs
            .iter()
            .for_each(|expr| get_definition(expr, definition_ass_list)),
        Expr::Local(local) => {
            !(ident_offset >= local.1.start && ident_offset < local.1.end)
        }
        Expr::List(lst) => {
            for expr in lst {
                match get_completion_of(expr, definition_map, ident_offset) {
                    true => continue,
                    false => return false,
                }
            }
            true
        }
    }
     */
    // TODO: implement this
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chumsky::{FormExpr, Func, ListExpr, LiteralExpr};
    use chumsky::chain::Chain;
    use chumsky::Span;

    #[test]
    pub fn can_complete_known_symbol() {
        // (defn foobar [x] (inc x))
        let body_le = Spanned::new(
            ListExpr::new(Spanned::new(
                FormsExpr::from(vec![Spanned::new(
                    FormExpr::from(Spanned::new(
                        LiteralExpr::Symbol(String::from("inc")),
                        18..21,
                    )),
                    18..21,
                )]),
                18..23,
            )),
            17..24,
        );
        let body = Spanned::new(
            FormsExpr::from(vec![Spanned::new(
                FormExpr::List(Box::new(body_le)),
                17..24,
            )]),
            17..24,
        );
        let mut ast = HashMap::new();
        ast.insert(
            String::from("foobar"),
            Func {
                args: vec![],
                body,
                name: ("foobar".to_string(), 6..12),
                span: 0..23,
            },
        );
        let item =
            ImCompleteCompletionItem::Function(String::from("fo"), vec![String::from("foobar")]);
        let actual = completion(&ast, 20);
        assert_eq!(1, actual.len());
        assert_eq!(vec!["foobar"], actual.keys().collect::<Vec<_>>());
        let actual_item: &ImCompleteCompletionItem = actual.get("foobar").unwrap();
        assert_eq!(
            &ImCompleteCompletionItem::Function(String::from("foobar"), vec![]),
            actual_item
        );
    }
}
