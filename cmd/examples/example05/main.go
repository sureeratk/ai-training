// This example shows you how to Ollama to create a reasonable response to a
// question with provided content.
//
// # Running the example:
//
//	$ make example5
//
// # This requires running the following command:
//
//  $ make ollama-up  // This starts the Ollama service.

package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
)

const (
	url   = "http://localhost:11434/v1/chat/completions"
	model = "qwen2.5vl:latest"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Minute)
	defer cancel()

	// Construct the http streaming client for interacting with the Ollama.
	cln := client.NewSSE[client.ChatSSE](client.StdoutLogger)

	// Format a prompt to direct the model what to do with the content and
	// the question.
	prompt := `Use the following pieces of information to answer the user's question.
	If you don't know the answer, say that you don't know.
	
	Context: %s
	
	Question: %s

	Answer the question and provide additional helpful information, but be concise.

	Responses should be properly formatted to be easily read.
	`

	content := `Intended Audience This notebook has been written and designed
	to provide a reference to everything that I say in the Ultimate Go class
	Its not necessarily a beginners Go book since it doesnt focus on the
	specifics of Gos syntax I would recommend the Go In Action book I wrote
	back in 2015 for that type of content Its still accurate and relevant
	Many of the things I say in the classroom over the 20 plus hours of
	instruction has been incorporated Ive tried to capture all the guidelines
	design philosophy whiteboarding and notes I share at the same moments I
	share them If you have taken the class before I believe this notebook will
	be invaluable for reminders on the content If you have never taken the class
	I still believe there is value in this book It covers more advanced topics
	not found in other books today Ive tried to provide a well rounded curriculum
	of topics from types to profiling I have also been able to provide examples
	for writing generic function and types in Go which will be available in
	version 118 of Go The book is written in the first person to drive home the
	idea that this is my book of notes from the Ultimate Go class The first
	chapter provides a set of design philosophies quotes and extra reading to
	help prepare your mind for the material Chapters 213 provide the core content
	from the class Chapter 14 provides a reediting of important blog posts Ive
	written in the past These posts are presented here to enhance some of the
	more technical chapters like garbage collection and concurrency If you are
	struggling with this book please provide me any feedback over email at`

	question := `Is there value in the book and why?`

	finalPrompt := fmt.Sprintf(prompt, content, question)

	d := client.D{
		"model": model,
		"messages": []client.D{
			{
				"role":    "user",
				"content": finalPrompt,
			},
		},
		"temperature": 1.0,
		"top_p":       0.5,
		"top_k":       20,
		"stream":      true,
	}

	ch := make(chan client.ChatSSE, 100)
	if err := cln.Do(ctx, http.MethodPost, url, d, ch); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	fmt.Print("Model Response:\n\n")

	for resp := range ch {
		fmt.Print(resp.Choices[0].Delta.Content)
	}

	return nil
}
