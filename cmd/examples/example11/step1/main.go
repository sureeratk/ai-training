// https://github.com/modelcontextprotocol/go-sdk
// https://github.com/modelcontextprotocol/go-sdk/blob/main/design/design.md
// https://github.com/orgs/modelcontextprotocol/discussions/364
//
// This example shows you how to create a basic MCP interaction where the Server
// is a CLI tool that hosts a set of tooling that is called by the Client for
// local machine interactions. This is what we need for example10 tooling.
//
// # Running the example:
//
//	$ make example11-step1
//
// # This doesn't require you to run any additional services.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func main() {
	cliMode := flag.Bool("cli", false, "run the program in cli mode")
	flag.Parse()

	if err := run(*cliMode); err != nil {
		log.Fatalln(err)
	}
}

func run(cliMode bool) error {
	if cliMode {
		return mcpTooling()
	}

	return testClient()
}

func mcpTooling() error {
	server := mcp.NewServer(&mcp.Implementation{Name: "file_lister", Version: "v1.0.0"}, nil)
	mcp.AddTool(server, &mcp.Tool{Name: "list_files", Description: "lists files"}, ListFiles)

	if err := server.Run(context.Background(), mcp.NewStdioTransport()); err != nil {
		return err
	}

	return nil
}

// =============================================================================

type ListFilesParams struct {
	Filter string `json:"filter" jsonschema:"a possible filter to use"`
}

func ListFiles(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[ListFilesParams]) (*mcp.CallToolResultFor[any], error) {
	data := struct {
		Filter string   `json:"filter"`
		Files  []string `json:"files"`
	}{
		Filter: params.Arguments.Filter,
		Files: []string{
			"file1.txt",
			"file2.txt",
			"file3.txt",
		},
	}

	d, err := json.Marshal(data)
	if err != nil {
		return nil, err
	}

	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(d),
		}},
	}, nil
}

// =============================================================================

func testClient() error {
	ctx := context.Background()

	client := mcp.NewClient(&mcp.Implementation{Name: "mcp-client", Version: "v1.0.0"}, nil)
	transport := mcp.NewCommandTransport(exec.Command(os.Args[0], "-cli", "true"))

	session, err := client.Connect(ctx, transport)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	params := &mcp.CallToolParams{
		Name:      "list_files",
		Arguments: map[string]any{"filter": "*.go"},
	}

	fmt.Printf("\nClient: Calling Tool: %s(%v)\n", params.Name, params.Arguments)

	res, err := session.CallTool(ctx, params)
	if err != nil {
		log.Fatalf("Tool Call FAILED: %v", err)
	}

	if res.IsError {
		log.Fatalf("Tool Call FAILED: %v", res.Content)
	}

	fmt.Print("Client: Waiting for Response\n\n")

	fmt.Println("Response:")
	for _, c := range res.Content {
		fmt.Print(c.(*mcp.TextContent).Text)
	}
	fmt.Print("\n")

	return nil
}
