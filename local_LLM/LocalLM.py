import ollama




class LocalLM:

    def __init__(self, model):
        # Initialize the Ollama client
        self.client = ollama.Client()
        self.model = model

    def get_llm_response(self, prompt):

        # Send the query to the model
        response = self.client.generate(model=self.model, prompt=prompt)
        return response.response
