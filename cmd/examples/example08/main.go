// This example shows you how to use the Llama3.2 model to generate SQL queries.
//
// # Running the example:
//
//	$ make openwebui
//    Use the OpenWebUI app with the Llama3.2:latest model.
//
// # This requires running the following commands:
//
//	$ make compose-up // This starts MongoDB and OpenWebUI in docker compose.
//  $ make ollama-up  // This starts the Ollama service.

package main

import (
	"bufio"
	"context"
	"database/sql"
	_ "embed"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"
	"time"

	"github.com/ardanlabs/ai-training/foundation/client"
	"github.com/ardanlabs/ai-training/foundation/sqldb"
	"github.com/jmoiron/sqlx"
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
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()

	db, err := dbInit(ctx)
	if err != nil {
		return fmt.Errorf("dbInit: %w", err)
	}

	defer db.Close()

	cln := client.New(client.StdoutLogger)

	// -------------------------------------------------------------------------

	reader := bufio.NewReader(os.Stdin)
	fmt.Print("\nAsk a question about the garage sale system: ")

	question, _ := reader.ReadString('\n')
	if question == "" {
		return nil
	}

	fmt.Print("\nGive me a second...\n\n")

	// -------------------------------------------------------------------------

	query, err := getQuery(ctx, cln, question)
	if err != nil {
		return fmt.Errorf("getQuery: %w", err)
	}

	fmt.Println("QUERY:")
	fmt.Print("-----------------------------------------------\n\n")
	fmt.Println(query)
	fmt.Print("\n")

	// -------------------------------------------------------------------------

	data := []map[string]any{}
	if err := sqldb.QueryMap(ctx, db, query, &data); err != nil {
		return fmt.Errorf("execQuery: %w", err)
	}

	fmt.Println("DATA:")
	fmt.Print("-----------------------------------------------\n\n")

	for i, m := range data {
		fmt.Printf("RESULT: %d\n", i+1)
		for k, v := range m {
			fmt.Printf("KEY: %s, VAL: %v\n", k, v)
		}
		fmt.Print("\n")
	}

	// -------------------------------------------------------------------------

	answer, err := getResponse(ctx, cln, question, data)
	if err != nil {
		return fmt.Errorf("getQuery: %w", err)
	}

	fmt.Println("ANSWER:")
	fmt.Print("-----------------------------------------------\n\n")
	fmt.Println(answer)
	fmt.Print("\n")

	return nil
}

var (
	//go:embed prompts/query.txt
	query string

	//go:embed prompts/response.txt
	response string
)

func getQuery(ctx context.Context, cln *client.Client, question string) (string, error) {
	d := client.D{
		"model": model,
		"messages": []client.D{
			{
				"role":    "user",
				"content": fmt.Sprintf(query, question),
			},
		},
		"temperature": 1.0,
		"top_p":       0.5,
		"top_k":       20,
	}

	var result client.Chat
	if err := cln.Do(ctx, http.MethodPost, url, d, &result); err != nil {
		return "", fmt.Errorf("do: %w", err)
	}

	return result.Choices[0].Message.Content, nil
}

func getResponse(ctx context.Context, cln *client.Client, question string, data []map[string]any) (string, error) {
	var builder strings.Builder
	for i, m := range data {
		builder.WriteString(fmt.Sprintf("RESULT: %d\n", i+1))
		for k, v := range m {
			builder.WriteString(fmt.Sprintf("KEY: %s, VAL: %v\n", k, v))
		}
		builder.WriteString("\n")
	}

	d := client.D{
		"model": model,
		"messages": []client.D{
			{
				"role":    "user",
				"content": fmt.Sprintf(response, builder.String(), question),
			},
		},
		"temperature": 1.0,
		"top_p":       0.5,
		"top_k":       20,
	}

	var result client.Chat
	if err := cln.Do(ctx, http.MethodPost, url, d, &result); err != nil {
		return "", fmt.Errorf("do: %w", err)
	}

	return result.Choices[0].Message.Content, nil
}

// =============================================================================

var (
	//go:embed sql/schema.sql
	schemaSQL string

	//go:embed sql/insert.sql
	insertSQL string
)

func dbInit(ctx context.Context) (*sqlx.DB, error) {
	fmt.Println("\nConnecting to the DB")

	db, err := dbConnection()
	if err != nil {
		return nil, fmt.Errorf("dbConnection: %w", err)
	}

	fmt.Println("Creating Schema")

	if err := dbExecute(ctx, db, schemaSQL); err != nil {
		return nil, fmt.Errorf("dbExecute: %w", err)
	}

	fmt.Println("Inserting Data")

	if err := dbExecute(ctx, db, insertSQL); err != nil {
		return nil, fmt.Errorf("dbExecute: %w", err)
	}

	return db, nil
}

func dbConnection() (*sqlx.DB, error) {
	db, err := sqldb.Open(sqldb.Config{
		User:         "postgres",
		Password:     "postgres",
		Host:         "localhost:5432",
		Name:         "postgres",
		MaxIdleConns: 0,
		MaxOpenConns: 0,
		DisableTLS:   true,
	})
	if err != nil {
		return nil, fmt.Errorf("connecting to db: %w", err)
	}

	return db, nil
}

func dbExecute(ctx context.Context, db *sqlx.DB, query string) error {
	if err := sqldb.StatusCheck(ctx, db); err != nil {
		return fmt.Errorf("status check database: %w", err)
	}

	tx, err := db.Begin()
	if err != nil {
		return err
	}

	defer func() {
		if errTx := tx.Rollback(); errTx != nil {
			if errors.Is(errTx, sql.ErrTxDone) {
				return
			}

			err = fmt.Errorf("rollback: %w", errTx)
			return
		}
	}()

	if _, err := tx.Exec(query); err != nil {
		return fmt.Errorf("exec: %w", err)
	}

	if err := tx.Commit(); err != nil {
		return fmt.Errorf("commit: %w", err)
	}

	return nil
}
