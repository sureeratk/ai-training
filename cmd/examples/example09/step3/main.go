// This example takes step2 and shows you how to store the image path and its
// vector embedding into a vector database.
//
// # Running the example:
//
//	$ make example9-step3
//
// # This requires running the following commands:
//
//	$ make ollama-up  // This starts the Ollama service.
//	$ make compose-up // This starts the MongoDB service.
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
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/mongodb"
	"go.mongodb.org/mongo-driver/bson"
	"go.mongodb.org/mongo-driver/mongo"
	"go.mongodb.org/mongo-driver/mongo/options"
)

const (
	urlChat        = "http://localhost:11434/v1/chat/completions"
	urlEmbedding   = "http://localhost:11434/v1/embeddings"
	modelChat      = "qwen2.5vl:latest"
	modelEmbedding = "bge-m3:latest"
	imagePath      = "cmd/samples/gallery/roseimg.png"
	dbName         = "example9"
	collectionName = "images-3"
)

type document struct {
	FileName    string    `bson:"file_name"`
	Description string    `bson:"description"`
	Embedding   []float64 `bson:"embedding"`
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

	cln := client.New(client.StdoutLogger)

	// -------------------------------------------------------------------------

	mongoClient, err := initDatabase(dbName, collectionName)
	if err != nil {
		return fmt.Errorf("initDatabase: %w", err)
	}

	col := mongoClient.Database(dbName).Collection(collectionName)

	// -------------------------------------------------------------------------

	findRes := col.FindOne(ctx, bson.D{{Key: "file_name", Value: imagePath}})
	if findRes.Err() == nil {
		fmt.Println("Delete existing image from database")
		_, err := col.DeleteOne(ctx, bson.D{{Key: "file_name", Value: imagePath}})
		if err != nil {
			return fmt.Errorf("delete image: %w", err)
		}
	}

	// -------------------------------------------------------------------------

	data, mimeType, err := readImage(imagePath)
	if err != nil {
		return fmt.Errorf("read image: %w", err)
	}

	// ---------------------------------------------------------------------

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

	// ---------------------------------------------------------------------

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

	// ---------------------------------------------------------------------

	fmt.Print("Inserting image description into the database:\n\n")

	d1 := document{
		FileName:    imagePath,
		Description: result.Choices[0].Message.Content,
		Embedding:   vector,
	}

	res, err := col.InsertOne(ctx, d1)
	if err != nil {
		return fmt.Errorf("insert: %w", err)
	}

	fmt.Printf("Inserted db id: %s\n\n", res.InsertedID)

	fmt.Println("DONE")
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

	fmt.Println("\nConnected to MongoDB")

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
