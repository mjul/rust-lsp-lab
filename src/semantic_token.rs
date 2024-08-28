use std::collections::HashMap;

use tower_lsp::lsp_types::SemanticTokenType;

use crate::chumsky::{
    FormExpr, FormsExpr, Func, ImCompleteSemanticToken, ListExpr, MapExpr, Spanned, VectorExpr,
};

pub const LEGEND_TYPE: &[SemanticTokenType] = &[
    SemanticTokenType::FUNCTION,
    SemanticTokenType::VARIABLE,
    SemanticTokenType::STRING,
    SemanticTokenType::COMMENT,
    SemanticTokenType::NUMBER,
    SemanticTokenType::KEYWORD,
    SemanticTokenType::OPERATOR,
    SemanticTokenType::PARAMETER,
];

/// Add semantic tokens from the AST.
/// Note that the parser also emits low-level semantic tokens for e.g. literals.
pub fn semantic_token_from_ast(ast: &HashMap<String, Func>) -> Vec<ImCompleteSemanticToken> {
    let mut semantic_tokens = vec![];

    ast.iter().for_each(|(_func_name, function)| {
        function.args.iter().for_each(|(_, span)| {
            semantic_tokens.push(ImCompleteSemanticToken {
                start: span.start,
                length: span.len(),
                token_type: LEGEND_TYPE
                    .iter()
                    .position(|item| item == &SemanticTokenType::PARAMETER)
                    .unwrap(),
            });
        });
        let (_, span) = &function.name;
        semantic_tokens.push(ImCompleteSemanticToken {
            start: span.start,
            length: span.len(),
            token_type: LEGEND_TYPE
                .iter()
                .position(|item| item == &SemanticTokenType::FUNCTION)
                .unwrap(),
        });
        semantic_token_from_forms_expr(&function.body, &mut semantic_tokens);
    });

    semantic_tokens
}

/// Recursively add semantic tokens from the forms to the `semantic_tokens` list.
fn semantic_token_from_forms_expr(
    expr: &Spanned<FormsExpr>,
    semantic_tokens: &mut Vec<ImCompleteSemanticToken>,
) {
    match &expr.0 {
        FormsExpr(bsfes) => {
            for fex in bsfes.iter() {
                semantic_token_from_form_expr(&fex, semantic_tokens);
            }
        }
    }
}

/// Recursively add semantic tokens from the form expression to the `semantic_tokens` list.
fn semantic_token_from_form_expr(
    expr: &Spanned<FormExpr>,
    semantic_tokens: &mut Vec<ImCompleteSemanticToken>,
) {
    match &expr.0 {
        FormExpr::Literal(_bsle) => {
            // Do nothing, literals are tagged in the parser
        }
        FormExpr::List(sles) => match &sles.0 {
            ListExpr(sfes) => {
                semantic_token_from_forms_expr(&sfes, semantic_tokens);
            }
        },
        FormExpr::Vector(bsve) => match &bsve.0 {
            VectorExpr(sfes) => {
                semantic_token_from_forms_expr(&sfes, semantic_tokens);
            }
        },
        FormExpr::Map(bsme) => match &bsme.0 {
            MapExpr(spairs) => {
                for (key, val) in spairs {
                    semantic_token_from_form_expr(key, semantic_tokens);
                    semantic_token_from_form_expr(val, semantic_tokens);
                }
            }
        },
    }
}
