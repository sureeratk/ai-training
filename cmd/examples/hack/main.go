package main

import (
	"context"
	"fmt"
	"log"

	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	llm, err := ollama.New(ollama.WithModel("llama3.2"))
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	llm.CallbacksHandler = callbacks.LogHandler{}

	var weather GetWeather
	tools := []llms.Tool{
		{
			Type: "function",
			Function: &llms.FunctionDefinition{
				Name:        weather.Name(),
				Description: weather.Description(),
				Parameters:  weather.Parameters(),
			},
		},
	}

	response, err := llm.GenerateContent(
		context.Background(),
		[]llms.MessageContent{
			llms.TextParts(llms.ChatMessageTypeHuman, "What is the weather in San Francisco, CA?"),
		},
		llms.WithTools(tools),
	)

	if err != nil {
		return fmt.Errorf("generate content: %w", err)
	}

	// Check if the model wants to call tools
	for _, choice := range response.Choices {
		for _, toolCall := range choice.ToolCalls {
			if toolCall.FunctionCall.Name == weather.Name() {

				// Execute the tool
				result, err := weather.Call(context.Background(), toolCall.FunctionCall.Arguments)
				if err != nil {
					return fmt.Errorf("tool call failed: %w", err)
				}
				fmt.Printf("Tool call result: %s\n", result)

				// Send result back to model for final response
				finalResponse, err := llm.GenerateContent(context.Background(),
					[]llms.MessageContent{
						llms.TextParts(llms.ChatMessageTypeHuman, "What is the weather in San Francisco, CA?"),
						llms.TextParts(llms.ChatMessageTypeAI, "I'll check the weather for you."),
						llms.TextParts(llms.ChatMessageTypeSystem, fmt.Sprintf("Weather data: %s", result)),
					})
				if err != nil {
					return fmt.Errorf("final response: %w", err)
				}

				fmt.Printf("Final response: %s\n", finalResponse.Choices[0].Content)
				return nil
			}
		}
	}

	fmt.Printf("Response: %s\n", response.Choices[0].Content)
	return nil
}

// =============================================================================

type GetWeather struct {
	Location string `json:"location"` // The city and state, e.g. San Francisco, CA
	Format   string `json:"format"`   // The format to return the weather in, e.g. 'celsius' or 'fahrenheit'
}

func (g GetWeather) Name() string {
	return "get_current_weather"
}

func (g GetWeather) Description() string {
	return "Get the current weather for a location"
}

func (g GetWeather) Parameters() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"location": map[string]any{
				"type":        "string",
				"description": "The location to get the weather for, e.g. San Francisco, CA",
			},
			"format": map[string]any{
				"type":        "string",
				"description": "The format to return the weather in, e.g. 'celsius' or 'fahrenheit'",
				"enum":        []string{"celsius", "fahrenheit"},
			},
		},
		"required": []string{"location", "format"},
	}
}

func (g GetWeather) Call(ctx context.Context, input string) (string, error) {
	return "hot and humid, 28 degrees celcius", nil
}
