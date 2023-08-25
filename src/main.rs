mod inference;

use crate::inference::ModelManager;
use rocket::{
    get,
    response::stream::TextStream,
    routes,
    serde::{json::Json, Deserialize, Serialize},
    State,
};
use std::sync::Mutex;

#[rocket::main]
async fn main() -> Result<(), rocket::Error> {
    let model_path = "C:\\Users\\Ethan\\Downloads\\llama-2-7b-chat.ggmlv3.q2_K.bin";

    let model = llm::load_dynamic(
        Some(llm::ModelArchitecture::Llama),
        std::path::Path::new(model_path),
        llm::TokenizerSource::Embedded,
        Default::default(),
        llm::load_progress_callback_stdout,
    )
    .expect(format!("Unable to load model at {}", model_path).as_ref());

    let session = model.start_session(Default::default());

    let manager = ModelManager { model, session };

    let _rocket = rocket::build()
        .manage(Mutex::new(manager))
        .mount("/", routes![health, complete])
        .launch()
        .await?;

    Ok(())
}

#[get("/health")]
async fn health() -> &'static str {
    "Service Ready to Accept Traffic!"
}

#[derive(Deserialize, Clone)]
#[serde(crate = "rocket::serde")]
struct CompletionRequest<'a> {
    prompt: &'a str,
}

#[derive(Serialize, Clone)]
#[serde(crate = "rocket::serde")]
struct CompletionUpdate {
    delta: String,
    err: Option<String>,
}

#[get("/complete", data = "<prompt>")]
async fn complete<'r>(
    _s: &'r State<Mutex<ModelManager>>,
    prompt: Json<CompletionRequest<'r>>,
) -> TextStream![String + 'r] {
    TextStream! {
        let up = CompletionUpdate {delta: prompt.prompt.into(), err: None};
        match serde_json::to_string(&up) {
            Ok(s) => yield format!("data: {}", s),
            Err(_) => {
                yield String::from(r#"data: {"delta":"", "err": "Error occured during generation}"#);
                return;
            },
        };
    }
}
