// This example takes step3 and shows you how to search for an image based on
// its description.
//
// # Running the example:
//
//	$ make example9-step4
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"github.com/dsoprea/go-exif/v3"
	exifcommon "github.com/dsoprea/go-exif/v3/common"
	jpg "github.com/dsoprea/go-jpeg-image-structure/v2"
	pis "github.com/dsoprea/go-png-image-structure/v2"
	"github.com/tmc/langchaingo/llms"
	"github.com/tmc/langchaingo/llms/ollama"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	url            = "http://localhost:11434"
	model          = "llama3.2-vision"
	imagePath      = "cmd/samples/roseimg.png"
	embedModel     = "bge-m3:latest"
	dbName         = "example9"
	collectionName = "images-4"
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

type document struct {
	FileName    string    `bson:"file_name"`
	Description string    `bson:"description"`
	Embedding   []float32 `bson:"embedding"`
}

type searchResult struct {
	FileName    string    `bson:"file_name" json:"file_name"`
	Description string    `bson:"description" json:"image_description"`
	Embedding   []float32 `bson:"embedding" json:"-"`
	Score       float64   `bson:"score" json:"-"`
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

	mongoClient, err := initDatabase(dbName, collectionName)
	if err != nil {
		return fmt.Errorf("initDatabase: %w", err)
	}

	col := mongoClient.Database(dbName).Collection(collectionName)

	// -------------------------------------------------------------------------

	data, mimeType, err := processImage(imagePath)
	if err != nil {
		return fmt.Errorf("process image: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Println("Generating image description...")

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

	if err := updateImage(imagePath, cr.Choices[0].Content); err != nil {
		return fmt.Errorf("update image: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Println("Generate embeddings for the image description")

	vectors, err := llmEmbed.CreateEmbedding(ctx, []string{cr.Choices[0].Content})
	if err != nil {
		return fmt.Errorf("create embedding: %w", err)
	}

	// -------------------------------------------------------------------------

	fmt.Printf("Inserting image description into the database: %s\n", cr.Choices[0].Content)

	if err := storeDocument(ctx, col, imagePath, cr.Choices[0].Content, vectors[0]); err != nil {
		return fmt.Errorf("storeDocument: %w", err)
	}

	// -------------------------------------------------------------------------

	reader := bufio.NewReader(os.Stdin)
	fmt.Print("Search the database for an image: ")

	question, _ := reader.ReadString('\n')
	if question == "" {
		return nil
	}

	fmt.Print("THIS MAY TAKE A MINUTE OR MORE, BE PATIENT\n\n")

	// -------------------------------------------------------------------------

	ctx, cancel := context.WithTimeout(context.Background(), 240*time.Second)
	defer cancel()

	results, err := vectorSearch(ctx, question)
	if err != nil {
		return fmt.Errorf("vectorSearch: %w", err)
	}

	if err := questionResponse(ctx, llm, question, results); err != nil {
		return fmt.Errorf("questionResponse: %w", err)
	}

	return nil
}

func initDatabase(dbName string, collectionName string) (*mongo.Client, error) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	// -------------------------------------------------------------------------
	// Connect to mongo

	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return nil, fmt.Errorf("connectToMongo: %w", err)
	}

	fmt.Println("Connected to MongoDB")

	// -------------------------------------------------------------------------
	// Create database and collection

	db := client.Database(dbName)

	col, err := mongodb.CreateCollection(ctx, db, collectionName)
	if err != nil {
		return nil, fmt.Errorf("createCollection: %w", err)
	}

	fmt.Println("Created Collection")

	// -------------------------------------------------------------------------
	// Create vector index

	const indexName = "vector_index"

	settings := mongodb.VectorIndexSettings{
		NumDimensions: 1024,
		Path:          "embedding",
		Similarity:    "cosine",
	}

	if err := mongodb.CreateVectorIndex(ctx, col, indexName, settings); err != nil {
		return nil, fmt.Errorf("createVectorIndex: %w", err)
	}

	fmt.Println("Created Vector Index")

	// -------------------------------------------------------------------------
	// Apply a unique index just to be safe.

	unique := true
	indexModel := mongo.IndexModel{
		Keys:    bson.D{{Key: "file_name", Value: 1}},
		Options: &options.IndexOptions{Unique: &unique},
	}
	if _, err := col.Indexes().CreateOne(ctx, indexModel); err != nil {
		return nil, fmt.Errorf("createUniqueIndex: %w", err)
	}

	fmt.Println("Created Unique file_name Index")

	return client, nil
}

func processImage(fileName string) ([]byte, string, error) {
	data, err := readImage(fileName)
	if err != nil {
		return nil, "", fmt.Errorf("read image: %w", err)
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

func storeDocument(ctx context.Context, col *mongo.Collection, fileName string, description string, vector []float32) error {

	// -------------------------------------------------------------------------
	// If this record already exist, we don't need to add it again.

	findRes := col.FindOne(ctx, bson.D{{Key: "file_name", Value: fileName}})
	switch {
	case findRes.Err() == nil:
		fmt.Println("Document already exists")
		return nil
	case !errors.Is(findRes.Err(), mongo.ErrNoDocuments):
		return fmt.Errorf("findOne: %w", findRes.Err())
	}

	// -------------------------------------------------------------------------
	// Let's add the document to the database.

	d1 := document{
		FileName:    fileName,
		Description: description,
		Embedding:   vector,
	}

	res, err := col.InsertOne(ctx, d1)
	if err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	fmt.Printf("Inserted db id: %s\n", res.InsertedID)

	return nil
}

func vectorSearch(ctx context.Context, question string) ([]searchResult, error) {

	// -------------------------------------------------------------------------
	// Use ollama to generate a vector embedding for the question.

	// Open a connection with ollama to access the model.
	llm, err := ollama.New(
		ollama.WithModel("bge-m3:latest"),
		ollama.WithServerURL("http://localhost:11434"),
	)
	if err != nil {
		return nil, fmt.Errorf("ollama: %w", err)
	}

	// Get the vector embedding for the question.
	embedding, err := llm.CreateEmbedding(context.Background(), []string{question})
	if err != nil {
		return nil, fmt.Errorf("create embedding: %w", err)
	}

	// -------------------------------------------------------------------------
	// Establish a connection with mongo and access the collection.

	// Connect to mongodb.
	client, err := mongodb.Connect(ctx, "mongodb://localhost:27017", "ardan", "ardan")
	if err != nil {
		return nil, fmt.Errorf("connectToMongo: %w", err)
	}

	const dbName = "example9"
	const collectionName = "images"

	// Capture a connection to the collection. We assume this exists with
	// data already.
	col := client.Database(dbName).Collection(collectionName)

	// -------------------------------------------------------------------------
	// Perform the vector search.

	// We want to find the nearest neighbors from the question vector embedding.
	pipeline := mongo.Pipeline{
		{{
			Key: "$vectorSearch",
			Value: bson.M{
				"index":         "vector_index",
				"exact":         false,
				"path":          "embedding",
				"queryVector":   embedding[0],
				"numCandidates": 5,
				"limit":         5,
			}},
		},
		{{
			Key: "$project",
			Value: bson.M{
				"file_name":   1,
				"description": 1,
				"embedding":   1,
				"score": bson.M{
					"$meta": "vectorSearchScore",
				},
			}},
		},
	}

	cur, err := col.Aggregate(ctx, pipeline)
	if err != nil {
		return nil, fmt.Errorf("aggregate: %w", err)
	}
	defer cur.Close(ctx)

	var results []searchResult
	if err := cur.All(ctx, &results); err != nil {
		return nil, fmt.Errorf("all: %w", err)
	}

	fmt.Println("=======================================")
	for _, result := range results {
		fmt.Printf("%s -> %.2f -> %s\n", result.FileName, result.Score, result.Description)
	}
	fmt.Print("=======================================\n")

	return results, nil
}

func questionResponse(ctx context.Context, llm *ollama.LLM, question string, results []searchResult) error {
	prompt := `
Use the following pieces of information to answer the user's question.
	
If you don't know the answer, say that you cannot find anything matching the description.	

Answer the question only with the full filename, including path, of the picture matching the description without providing any additional details except what you already have.

The response should be in a JSON format with the following fields:
{"status": "found", "filename": "<filename>"}

If the file is missing, we should have this response:
{"status": "not found"}

Responses should be properly formatted and always a JSON like in the example.
Make sure the path of the file is always the same as that specified in the context.
Do not add anything to the path if the path is relative or not a fully qualified path.
Ensure that output path is the one in the input path and matches every character.

The data in the context is a JSON object with the following fields:
[
	{"file_name":"<filepath>", "image_description":"<description>"},
]

Context:
%s
	
Question: %s
`
	content, err := json.MarshalIndent(results, "", "  ")
	if err != nil {
		return fmt.Errorf("marshal: %w", err)
	}

	finalPrompt := fmt.Sprintf(prompt, string(content), question)

	fmt.Print("\n=====================================\n")
	fmt.Println(finalPrompt)
	fmt.Print("\n======================================\n")

	// This function will display the response as it comes from the server.
	f := func(ctx context.Context, chunk []byte) error {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		fmt.Printf("%s", chunk)
		return nil
	}

	// Send the prompt to the model server.
	_, err = llm.Call(
		ctx,
		finalPrompt,
		llms.WithStreamingFunc(f),
		llms.WithMaxTokens(contextWindow),
	)
	if err != nil {
		return fmt.Errorf("call: %w", err)
	}

	return nil
}
