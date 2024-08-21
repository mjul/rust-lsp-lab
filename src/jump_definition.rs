use std::collections::HashMap;

use im_rc::Vector;

use crate::chumsky::{Expr, FormsExpr, Func, Spanned};
/// return (need_to_continue_search, founded reference)
pub fn get_definition(ast: &HashMap<String, Func>, ident_offset: usize) -> Option<Spanned<String>> {
    let mut vector = Vector::new();
    for (_, v) in ast.iter() {
        if v.name.1.start < ident_offset && v.name.1.end > ident_offset {
            return Some(v.name.clone());
        }
        if v.name.1.end < ident_offset {
            vector.push_back(v.name.clone());
        }
    }

    for (_, v) in ast.iter() {
        let args = v.args.iter().cloned().collect::<Vector<_>>();
        if let (_, Some(value)) =
            get_definition_of_expr(&v.body, args + vector.clone(), ident_offset)
        {
            return Some(value);
        }
    }
    None
}

pub fn get_definition_of_expr(
    expr: &Spanned<FormsExpr>,
    definition_ass_list: Vector<Spanned<String>>,
    ident_offset: usize,
) -> (bool, Option<Spanned<String>>) {
    /*
    match &expr.0 {
        Expr::Error => (true, None),
        Expr::Value(_) => (true, None),
        Expr::List(exprs) => exprs
            .iter()
            .for_each(|expr| get_definition(expr, definition_ass_list)),
        Expr::Local(local) => {
            if ident_offset >= local.1.start && ident_offset < local.1.end {
                let index = definition_ass_list
                    .iter()
                    .position(|decl| decl.0 == local.0);
                (
                    false,
                    index.map(|i| definition_ass_list.get(i).unwrap().clone()),
                )
            } else {
                (true, None)
            }
        }
        Expr::List(lst) => {
            for expr in lst {
                match get_definition_of_expr(expr, definition_ass_list.clone(), ident_offset) {
                    (true, None) => continue,
                    (true, Some(value)) => return (false, Some(value)),
                    (false, None) => return (false, None),
                    (false, Some(value)) => return (false, Some(value)),
                }
            }
            (true, None)
        }
    }*/
    // TODO: implement this
    (true, None)
}
