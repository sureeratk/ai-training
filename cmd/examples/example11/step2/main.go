// https://github.com/modelcontextprotocol/go-sdk
// https://github.com/modelcontextprotocol/go-sdk/blob/main/design/design.md
// https://github.com/orgs/modelcontextprotocol/discussions/364
//
// This example shows you how to create a basic MCP interaction where the Server
// runs as a service and extends the set of tools as endpoints. The Client makes
// a call to the Server via the MCP SSE protocol. The makefile shows you the
// raw CURL calls that are used to make the client/server interaction.
//
// # Running the example:
//
//	$ make example11-step2
//
// # This doesn't require you to run any additional services.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/signal"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func main() {
	host := flag.String("host", "localhost", "host to listen on")
	port := flag.String("port", "8080", "port to listen on")
	flag.Parse()

	if err := run(*host, *port); err != nil {
		log.Fatal(err)
	}
}

func run(host string, port string) error {
	go func() {
		server(host, port)
	}()

	fmt.Println("\nTesting MCP Client coded against the MCP Server")

	if err := client(host, port); err != nil {
		return err
	}

	fmt.Print("\nTest Successful\n\n")

	fmt.Println("Holding the server open for extended testing.\n\nPress Ctrl+C to exit.")

	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt)
	<-ch

	fmt.Println("\nServer Down")

	return nil
}

// =============================================================================
// Tools

type ListFilesParams struct {
	Filter string `json:"filter" jsonschema:"a possible filter to use"`
}

func ListFiles(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[ListFilesParams]) (*mcp.CallToolResultFor[any], error) {
	data := struct {
		Status string   `json:"status"`
		Filter string   `json:"filter"`
		Files  []string `json:"files"`
	}{
		Status: "SUCCESS",
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
// Basic server implementation

func server(host string, port string) {
	fileLister := mcp.NewServer(&mcp.Implementation{Name: "file_lister", Version: "v1.0.0"}, nil)
	mcp.AddTool(fileLister, &mcp.Tool{Name: "list_files", Description: "lists files"}, ListFiles)

	// -------------------------------------------------------------------------

	addr := fmt.Sprintf("%s:%s", host, port)
	log.Printf("Server: MCP servers serving at %s", addr)

	// -------------------------------------------------------------------------

	f := func(request *http.Request) *mcp.Server {
		url := request.URL.Path
		log.Printf("Server: Handling request for URL %s\n", url)

		switch url {
		case "/list_files":
			return fileLister

		default:
			return mcp.NewServer(&mcp.Implementation{Name: "unknown_tool", Version: "v1.0.0"}, nil)
		}
	}

	handler := mcp.NewSSEHandler(f)
	log.Fatal(http.ListenAndServe(addr, handler))
}

// =============================================================================
// Basic client code

func client(host string, port string) error {
	ctx := context.Background()

	addr := fmt.Sprintf("http://%s:%s/list_files", host, port)

	client := mcp.NewClient(&mcp.Implementation{Name: "mcp-client", Version: "v1.0.0"}, nil)

	transport := mcp.NewSSEClientTransport(addr, nil)

	fmt.Print("Client: Connecting to MCP Server\n\n")

	session, err := client.Connect(ctx, transport)
	if err != nil {
		return fmt.Errorf("failed to connect to MCP server: %w", err)
	}
	defer session.Close()

	params := &mcp.CallToolParams{
		Name:      "list_files",
		Arguments: map[string]any{"filter": "*.go"},
	}

	fmt.Printf("\nClient: Calling Tool: %s(%v)\n", params.Name, params.Arguments)

	res, err := session.CallTool(ctx, params)
	if err != nil {
		return fmt.Errorf("failed to call tool: %w", err)
	}

	if res.IsError {
		return fmt.Errorf("tool call failed: %s", res.Content)
	}

	fmt.Println("Client: Waiting for Response")

	for _, c := range res.Content {
		fmt.Print(c.(*mcp.TextContent).Text)
	}
	fmt.Print("\n")

	return nil
}
