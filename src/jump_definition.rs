use im_rc::Vector;
use std::collections::HashMap;

use crate::chumsky::{Defn, FormExpr, FormsExpr, Func, ListExpr, Spanned, VectorExpr};

/// Try to find a the span with the function name definition corresponding
/// to the symbol at the cursor position, [`ident_offset`].
/// return (need_to_continue_search, the found reference if any)
pub fn get_definition(ast: &HashMap<String, Func>, ident_offset: usize) -> Option<Spanned<String>> {
    let mut names_defined_before_the_ident_offset = Vector::new();
    for (_, v) in ast.iter() {
        if v.name.1.start < ident_offset && v.name.1.end > ident_offset {
            // if the offset is inside the function definition name span that name is the answer
            return Some(v.name.clone());
        }
        if v.name.1.end < ident_offset {
            names_defined_before_the_ident_offset.push_back(v.name.clone());
        }
    }

    for (_, v) in ast.iter() {
        let args = v.args.iter().cloned().collect::<Vector<_>>();
        if let (_, Some(value)) = get_definition_of_forms_expr(
            &v.body,
            args + names_defined_before_the_ident_offset.clone(),
            ident_offset,
        ) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chumsky::{LiteralExpr, Spanned};
    use chumsky::Span;

    fn just_defn_foobar() -> HashMap<String, Func> {
        // (defn foobar [x] (inc x))
        let foobar_body_le = Spanned::new(
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
        let foobar_body = Spanned::new(
            FormsExpr::from(vec![Spanned::new(
                FormExpr::List(Box::new(foobar_body_le)),
                17..24,
            )]),
            17..24,
        );
        let mut ast = HashMap::new();
        ast.insert(
            String::from("foobar"),
            Func {
                args: vec![],
                body: foobar_body,
                name: ("foobar".to_string(), 6..12),
                span: 0..23,
            },
        );
        ast
    }

    #[test]
    pub fn can_get_definition_for_function_when_cursor_is_inside_function_name() {
        let ast = just_defn_foobar();
        let actual = get_definition(&ast, 8);
        assert_eq!(Some(Spanned::new(String::from("foobar"), 6..12)), actual);
    }
}
