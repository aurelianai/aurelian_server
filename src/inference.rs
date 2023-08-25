use std::convert::Infallible;

pub struct ModelManager {
    pub model: Box<dyn llm::Model>,
    pub session: llm::InferenceSession,
}

impl ModelManager {
    pub fn infer<F>(&mut self, prompt: &str, callback: F) -> Result<llm::InferenceStats, String>
    where
        F: FnMut(llm::InferenceResponse) -> Result<llm::InferenceFeedback, Infallible>,
    {
        self.session
            .infer(
                self.model.as_ref(),
                &mut rand::thread_rng(),
                &llm::InferenceRequest {
                    prompt: llm::Prompt::Text(prompt),
                    parameters: &llm::InferenceParameters::default(),
                    play_back_previous_tokens: false,
                    maximum_token_count: None,
                },
                &mut Default::default(),
                callback,
            )
            .map_err(|e| e.to_string())
    }
}
