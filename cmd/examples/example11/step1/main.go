package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func main() {
	runServer := flag.Bool("server", false, "run in server mode")
	flag.Parse()

	if err := run(*runServer); err != nil {
		log.Fatalln(err)
	}
}

func run(runServer bool) error {
	if runServer {
		server()
		return nil
	}

	if err := client(); err != nil {
		return err
	}

	return nil
}

// =============================================================================

type HiParams struct {
	Name string `json:"name" jsonschema:"the name of the person to greet"`
}

func SayHi(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[HiParams]) (*mcp.CallToolResultFor[any], error) {
	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{&mcp.TextContent{Text: "Server: Hi " + params.Arguments.Name}},
	}, nil
}

func server() {
	server := mcp.NewServer(&mcp.Implementation{Name: "greeter", Version: "v1.0.0"}, nil)

	mcp.AddTool(server, &mcp.Tool{Name: "greet", Description: "say hi"}, SayHi)

	if err := server.Run(context.Background(), mcp.NewStdioTransport()); err != nil {
		log.Fatal(err)
	}
}

// =============================================================================

func client() error {
	ctx := context.Background()

	client := mcp.NewClient(&mcp.Implementation{Name: "mcp-client", Version: "v1.0.0"}, nil)

	transport := mcp.NewCommandTransport(exec.Command(os.Args[0], "-server", "true"))
	session, err := client.Connect(ctx, transport)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	params := &mcp.CallToolParams{
		Name:      "greet",
		Arguments: map[string]any{"name": "you"},
	}

	fmt.Printf("Client: CALLING TOOL: %s(%v)\n", params.Name, params.Arguments)

	res, err := session.CallTool(ctx, params)
	if err != nil {
		log.Fatalf("CallTool failed: %v", err)
	}

	if res.IsError {
		log.Fatal("tool failed")
	}

	fmt.Println("Client: Waiting for Response")

	for _, c := range res.Content {
		fmt.Print(c.(*mcp.TextContent).Text)
	}
	fmt.Print("\n")

	fmt.Println("Client: Done")

	return nil
}
