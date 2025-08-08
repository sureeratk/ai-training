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
		log.Fatalln(err)
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

	fmt.Print("Test Successful\n\n")

	fmt.Println("Holding the server open for extended testing.\n\nPress Ctrl+C to exit.")

	ch := make(chan os.Signal, 1)
	signal.Notify(ch, os.Interrupt)
	<-ch

	fmt.Println("\nServer Down")

	return nil
}

// =============================================================================

var text = `While it may not be obvious to everyone, there are a number of reasons creating random paragraphs can be useful. A few examples of how some people use this generator are listed in the following paragraphs.`

type HiParams struct {
	Name string `json:"name" jsonschema:"the name of the person to greet"`
}

func SayHi(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[HiParams]) (*mcp.CallToolResultFor[any], error) {
	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{&mcp.TextContent{Text: "Server: Hi " + params.Arguments.Name + "\n" + text}},
	}, nil
}

func SayError(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[any]) (*mcp.CallToolResultFor[any], error) {
	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{&mcp.TextContent{Text: "Server: Error"}},
	}, nil
}

func server(host string, port string) {
	server1 := mcp.NewServer(&mcp.Implementation{Name: "greeter1"}, nil)
	mcp.AddTool(server1, &mcp.Tool{Name: "greet1", Description: "say hi"}, SayHi)

	server2 := mcp.NewServer(&mcp.Implementation{Name: "greeter2"}, nil)
	mcp.AddTool(server2, &mcp.Tool{Name: "greet2", Description: "say hello"}, SayHi)

	server3 := mcp.NewServer(&mcp.Implementation{Name: "error"}, nil)
	mcp.AddTool(server3, &mcp.Tool{Name: "error", Description: "error"}, SayError)

	addr := fmt.Sprintf("%s:%s", host, port)

	log.Printf("Server: MCP servers serving at %s", addr)

	f := func(request *http.Request) *mcp.Server {
		url := request.URL.Path
		log.Printf("Server: Handling request for URL %s\n", url)

		switch url {
		case "/greet1":
			return server1

		case "/greet2":
			return server2

		default:
			return server3
		}
	}

	handler := mcp.NewSSEHandler(f)

	log.Fatal(http.ListenAndServe(addr, handler))
}

// =============================================================================

func client(host string, port string) error {
	ctx := context.Background()

	addr := fmt.Sprintf("http://%s:%s/greet1", host, port)

	client := mcp.NewClient(&mcp.Implementation{Name: "mcp-client", Version: "v1.0.0"}, nil)

	transport := mcp.NewSSEClientTransport(addr, nil)

	fmt.Println("Client: Connecting to MCP Server")

	session, err := client.Connect(ctx, transport)
	if err != nil {
		log.Fatal(err)
	}
	defer session.Close()

	params := &mcp.CallToolParams{
		Name:      "greet1",
		Arguments: map[string]any{"name": "you"},
	}

	fmt.Printf("Client: Calling Tool: %s(%v)\n", params.Name, params.Arguments)

	res, err := session.CallTool(ctx, params)
	if err != nil {
		log.Fatalf("Tool Call FAILED: %v", err)
	}

	if res.IsError {
		log.Fatalf("Tool Call FAILED: %v", res.Content)
	}

	fmt.Println("Client: Waiting for Response")

	for _, c := range res.Content {
		fmt.Print(c.(*mcp.TextContent).Text)
	}
	fmt.Print("\n")

	fmt.Println("Client: Done")

	return nil
}
