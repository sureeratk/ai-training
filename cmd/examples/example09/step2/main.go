// This examples takes step1 and shows you how to generate a vector embedding
// from the image description.
//
// # Running the example:
//
//	$ make example9-step2
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
package main

import (
	"context"
	"encoding/base64"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"

	"github.com/ardanlabs/ai-training/foundation/client"
)

const (
	urlChat        = "http://localhost:11434/v1/chat/completions"
	urlEmbedding   = "http://localhost:11434/v1/embeddings"
	modelChat      = "qwen2.5vl:latest"
	modelEmbedding = "bge-m3:latest"
	imagePath      = "cmd/samples/gallery/roseimg.png"
)

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	// -------------------------------------------------------------------------

	cln := client.New(client.StdoutLogger)

	// -------------------------------------------------------------------------

	data, mimeType, err := readImage(imagePath)
	if err != nil {
		return fmt.Errorf("read image: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Print("\nGenerating image description:\n\n")

	prompt := `Describe the image. Be concise and accurate. Do not be overly
	verbose or stylistic. Make sure all the elements in the image are
	enumerated and described. Do not include any additional details. Keep
	the description under 200 words. At the end of the description, create
	a list of tags with the names of all the elements in the image. Do not
	output anything past this list.
	Encode the list as valid JSON, as in this example:
	[
		"tag1",
		"tag2",
		"tag3",
		...
	]
	Make sure the JSON is valid, doesn't have any extra spaces, and is
	properly formatted.`

	dataBase64 := base64.StdEncoding.EncodeToString(data)

	d := client.D{
		"model": modelChat,
		"messages": []client.D{
			{
				"role": "user",
				"content": []client.D{
					{
						"type": "text",
						"text": prompt,
					},
					{
						"type": "image_url",
						"image_url": client.D{
							"url": fmt.Sprintf("data:%s;base64,%s", mimeType, dataBase64),
						},
					},
				},
			},
		},
		"temperature": 1.0,
		"top_p":       0.5,
		"top_k":       20,
	}

	var result client.Chat
	if err := cln.Do(ctx, http.MethodPost, urlChat, d, &result); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	fmt.Print(result.Choices[0].Message.Content)
	fmt.Print("\n\n")

	// -------------------------------------------------------------------------

	fmt.Print("Generate embeddings for the image description:\n\n")

	d = client.D{
		"model":              modelEmbedding,
		"truncate":           true,
		"truncate_direction": "right",
		"input":              result.Choices[0].Message.Content,
	}

	// Get the vector embedding for this question.
	var resp client.Embedding
	if err := cln.Do(ctx, http.MethodPost, urlEmbedding, d, &resp); err != nil {
		return fmt.Errorf("do: %w", err)
	}

	vector := resp.Data[0].Embedding

	fmt.Printf("%v...%v\n\n", vector[0:3], vector[len(vector)-3:])

	fmt.Println("DONE")
	return nil
}

func readImage(fileName string) ([]byte, string, error) {
	f, err := os.OpenFile(fileName, os.O_RDONLY, 0)
	if err != nil {
		return nil, "", fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, "", fmt.Errorf("read file: %w", err)
	}

	var mimeType string
	switch filepath.Ext(fileName) {
	case ".jpg", ".jpeg":
		mimeType = "image/jpg"
	case ".png":
		mimeType = "image/png"
	default:
		return nil, "", fmt.Errorf("unsupported file type: %s", filepath.Ext(fileName))
	}

	return data, mimeType, nil
}
