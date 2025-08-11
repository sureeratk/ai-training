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
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"

	"github.com/dsoprea/go-exif/v3"
	exifcommon "github.com/dsoprea/go-exif/v3/common"
	jpg "github.com/dsoprea/go-jpeg-image-structure/v2"
	pis "github.com/dsoprea/go-png-image-structure/v2"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
)

const (
	url        = "http://localhost:11434"
	model      = "llama3.2-vision"
	embedModel = "bge-m3:latest"
)

// The context window represents the maximum number of tokens that can be sent
// and received by the model. The default for Ollama is 8K. In the makefile
// it has been increased to 64K.
var contextWindow = 1024 * 8

func init() {
	if v := os.Getenv("OLLAMA_CONTEXT_LENGTH"); v != "" {
		var err error
		contextWindow, err = strconv.Atoi(v)
		if err != nil {
			log.Fatal(err)
		}
	}
}

// =============================================================================

func main() {
	if err := run(); err != nil {
		log.Fatal(err)
	}
}

func run() error {
	ctx := context.Background()

	// -------------------------------------------------------------------------
	// Connect to Ollama

	llm, err := ollama.New(
		ollama.WithModel(model),
		ollama.WithServerURL(url),
	)
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	llmEmbed, err := ollama.New(
		ollama.WithModel(embedModel),
		ollama.WithServerURL(url),
	)
	if err != nil {
		return fmt.Errorf("ollama: %w", err)
	}

	// -------------------------------------------------------------------------

	fileName := "cmd/samples/roseimg.png"

	data, err := readImage(fileName)
	if err != nil {
		return fmt.Errorf("read image: %w", err)
	}

	prompt := `Describe the image.
Be concise and accurate.
Do not be overly verbose or stylistic.
Make sure all the elements in the image are enumerated and described.
Do not include any additional details.
Keep the description under 200 words.
At the end of the description, create a list of tags with the names of all the elements in the image.
Do no output anything past this list.
Encode the list as valid JSON, as in this example:
[
  "tag1",
  "tag2",
  "tag3",
  ...
]
Make sure the JSON is valid, doesn't have any extra spaces, and is properly formatted.
`

	var mimeType string
	switch filepath.Ext(fileName) {
	case ".jpg", ".jpeg":
		mimeType = "image/jpg"
	case ".png":
		mimeType = "image/png"
	default:
		return fmt.Errorf("unsupported file type: %s", filepath.Ext(fileName))
	}

	// -------------------------------------------------------------------------

	fmt.Println("Generating image description...")

	messages := []llms.MessageContent{
		{
			Role: llms.ChatMessageTypeHuman,
			Parts: []llms.ContentPart{
				llms.BinaryContent{
					MIMEType: mimeType,
					Data:     data,
				},
				llms.TextContent{
					Text: prompt,
				},
			},
		},
	}

	cr, err := llm.GenerateContent(
		ctx,
		messages,
		llms.WithMaxTokens(contextWindow),
	)
	if err != nil {
		return fmt.Errorf("generate content: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Printf("Updating Image description: %s\n", cr.Choices[0].Content)

	err = updateImage(fileName, cr.Choices[0].Content)
	if err != nil {
		return fmt.Errorf("update image: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Println("Generate embeddings for the image description")

	vectors, err := llmEmbed.CreateEmbedding(ctx, []string{cr.Choices[0].Content})
	if err != nil {
		return fmt.Errorf("create embedding: %w", err)
	}

	fmt.Printf("Received embeddings from model: %v\n", vectors[0])

	return nil
}

func readImage(fileName string) ([]byte, error) {
	f, err := os.OpenFile(fileName, os.O_RDONLY, 0)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	data, err := io.ReadAll(f)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	return data, nil
}

func updateImage(fileName string, description string) error {
	im, err := exifcommon.NewIfdMappingWithStandard()
	if err != nil {
		return fmt.Errorf("new idf mapping: %w", err)
	}

	ti := exif.NewTagIndex()
	ib := exif.NewIfdBuilder(im, ti, exifcommon.IfdStandardIfdIdentity, exifcommon.EncodeDefaultByteOrder)

	err = ib.AddStandardWithName("ImageDescription", description)
	if err != nil {
		return fmt.Errorf("add standard: %w", err)
	}

	// -------------------------------------------------------------------------

	switch filepath.Ext(fileName) {
	case ".jpg", ".jpeg":
		intfc, err := jpg.NewJpegMediaParser().ParseFile(fileName)
		if err != nil {
			return fmt.Errorf("parse file: %w", err)
		}

		cs := intfc.(*jpg.SegmentList)
		err = cs.SetExif(ib)
		if err != nil {
			return fmt.Errorf("set ib: %w", err)
		}

		f, err := os.Create(fileName)
		if err != nil {
			return fmt.Errorf("create: %w", err)
		}

		err = cs.Write(f)
		if err != nil {
			return fmt.Errorf("write: %w", err)
		}
		defer f.Close()

	case ".png":
		intfc, err := pis.NewPngMediaParser().ParseFile(fileName)
		if err != nil {
			return fmt.Errorf("parse file: %w", err)
		}

		cs := intfc.(*pis.ChunkSlice)
		err = cs.SetExif(ib)
		if err != nil {
			return fmt.Errorf("set ib: %w", err)
		}

		f, err := os.Create(fileName)
		if err != nil {
			return fmt.Errorf("create: %w", err)
		}

		err = cs.WriteTo(f)
		if err != nil {
			return fmt.Errorf("write: %w", err)
		}
		defer f.Close()

	default:
		return fmt.Errorf("unsupported file type: %s", filepath.Ext(fileName))
	}

	return nil
}
