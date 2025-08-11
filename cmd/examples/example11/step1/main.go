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
//	$ make example11-step1
//
// # This doesn't require you to run any additional services.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"os/exec"
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
		mcpListenAndServe(host, port)
	}()

	fmt.Println("\nTesting MCP Client coded against the MCP Server")

	if err := client(host, port, "list_files", map[string]any{"filter": "*.go"}); err != nil {
		return err
	}

	if err := client(host, port, "read_files", map[string]any{"path": "file1.txt"}); err != nil {
		return err
	}

	if err := client(host, port, "shell_command", map[string]any{"command": []string{"find", ".", "-name", "*.go", "-not", "-path", "./vendor/*"}}); err != nil {
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

type ReadFilesParams struct {
	Path string `json:"path" jsonschema:"the path to the file to read"`
}

func ReadFiles(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[ReadFilesParams]) (*mcp.CallToolResultFor[any], error) {
	data := struct {
		Status  string `json:"status"`
		Path    string `json:"path"`
		Content string `json:"content"`
	}{
		Status:  "SUCCESS",
		Path:    params.Arguments.Path,
		Content: "Hello World",
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

type ShellCommandParams struct {
	Command []string `json:"command" jsonschema:"the command and arguments to execute"`
}

// ShellCommand is a VERY DANGEROUS tool that should never be implemented like this.
// I am showing this because you could leverage CLI tooling to do things like
// list files, read files, etc, but you need some way to limit the commands that
// can be executed with a level of security.
func ShellCommand(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[ShellCommandParams]) (*mcp.CallToolResultFor[any], error) {
	var out bytes.Buffer
	cmd := exec.Command(params.Arguments.Command[0], params.Arguments.Command[1:]...)
	cmd.Stdout = &out
	if err := cmd.Run(); err != nil {
		return nil, err
	}

	data := struct {
		Command []string `json:"command"`
		Output  string   `json:"output"`
	}{
		Command: params.Arguments.Command,
		Output:  out.String(),
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

func mcpListenAndServe(host string, port string) {
	fileOperations := mcp.NewServer(&mcp.Implementation{Name: "file operations", Version: "v1.0.0"}, nil)
	mcp.AddTool(fileOperations, &mcp.Tool{Name: "list_files", Description: "lists files"}, ListFiles)
	mcp.AddTool(fileOperations, &mcp.Tool{Name: "read_files", Description: "reads a file"}, ReadFiles)
	mcp.AddTool(fileOperations, &mcp.Tool{Name: "shell_command", Description: "runs a shell command"}, ShellCommand)

	// -------------------------------------------------------------------------

	addr := fmt.Sprintf("%s:%s", host, port)
	log.Printf("Server: MCP servers serving at %s", addr)

	// -------------------------------------------------------------------------

	f := func(request *http.Request) *mcp.Server {
		url := request.URL.Path
		log.Printf("Server: Handling request for URL %s\n", url)

		switch url {
		case "/list_files":
			return fileOperations

		case "/read_files":
			return fileOperations

		case "/shell_command":
			return fileOperations

		default:
			return mcp.NewServer(&mcp.Implementation{Name: "unknown_tool", Version: "v1.0.0"}, nil)
		}
	}

	handler := mcp.NewSSEHandler(f)
	log.Fatal(http.ListenAndServe(addr, handler))
}

// =============================================================================
// Basic client code

func client(host string, port string, tool string, arguments map[string]any) error {
	ctx := context.Background()

	addr := fmt.Sprintf("http://%s:%s/%s", host, port, tool)

	client := mcp.NewClient(&mcp.Implementation{Name: "mcp-client", Version: "v1.0.0"}, nil)

	transport := mcp.NewSSEClientTransport(addr, nil)

	fmt.Print("Client: Connecting to MCP Server\n\n")

	session, err := client.Connect(ctx, transport)
	if err != nil {
		return fmt.Errorf("failed to connect to MCP server: %w", err)
	}
	defer session.Close()

	params := &mcp.CallToolParams{
		Name:      tool,
		Arguments: arguments,
	}

	fmt.Printf("\nClient: Calling Tool: %s(%v)\n\n", params.Name, params.Arguments)

	res, err := session.CallTool(ctx, params)
	if err != nil {
		return fmt.Errorf("failed to call tool: %w", err)
	}

	if res.IsError {
		return fmt.Errorf("tool call failed: %s", res.Content)
	}

	for _, c := range res.Content {
		fmt.Print(c.(*mcp.TextContent).Text)
	}
	fmt.Print("\n")

	return nil
}
