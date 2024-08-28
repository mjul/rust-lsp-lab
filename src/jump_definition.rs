use im_rc::Vector;
use std::collections::HashMap;

use crate::chumsky::{Defn, FormExpr, FormsExpr, Func, ListExpr, Spanned, VectorExpr};
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
            get_definition_of_forms_expr(&v.body, args + vector.clone(), ident_offset)
        {
            return Some(value);
        }
    }
    None
}

/// return (need_to_continue_search, founded reference)
fn get_definition_of_forms_expr(
    expr: &Spanned<FormsExpr>,
    definition_ass_list: Vector<Spanned<String>>,
    ident_offset: usize,
) -> (bool, Option<Spanned<String>>) {
    match &expr.0 {
        FormsExpr(bsfes) => {
            for fex in bsfes.iter() {
                let (cont, def) =
                    get_definition_of_form_expr(&fex, definition_ass_list.clone(), ident_offset);
                match (cont, def) {
                    (false, def) => return (false, def),
                    _ => {
                        // TODO
                    }
                }
            }
        }
    }

    // TODO: implement this
    (true, None)
}

/// return (need_to_continue_search, founded reference)
fn get_definition_of_form_expr(
    expr: &Spanned<FormExpr>,
    definition_ass_list: Vector<Spanned<String>>,
    ident_offset: usize,
) -> (bool, Option<Spanned<String>>) {
    match &expr.0 {
        FormExpr::Literal(_bsle) => (true, None),
        FormExpr::List(sles) => match &sles.0 {
            ListExpr(sfes) => {
                // TODO: introduce a higher-level semantical expression with e.g. def, defn and let-bound symbols so we don't have to parse here in the client code
                // Handle defn
                let result = Defn::try_from(expr.0.clone());
                match result {
                    Ok(defn) => (false, Some(defn.name)),
                    Err(_) => {
                        // Not a defn, check def and let or recurse
                        // TODO: def and let
                        // Otherwise, not interesting in itself, search the list elements:
                        get_definition_of_forms_expr(
                            sfes,
                            definition_ass_list.clone(),
                            ident_offset,
                        )
                    }
                }
            }
        },
        FormExpr::Vector(bsve) => match &bsve.0 {
            VectorExpr(_sfes) => (true, None),
        },
        FormExpr::Map(_bsme) => (true, None),
    }
}
