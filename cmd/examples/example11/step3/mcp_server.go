package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

func server(host string, port string) {
	fileOperations := mcp.NewServer(&mcp.Implementation{Name: "file_operations", Version: "v1.0.0"}, nil)
	mcp.AddTool(fileOperations, &mcp.Tool{Name: "read_file", Description: "reads a file"}, MCPReadFile)

	// -------------------------------------------------------------------------

	addr := fmt.Sprintf("%s:%s", host, port)
	log.Printf("Server: MCP servers serving at %s", addr)

	// -------------------------------------------------------------------------

	f := func(request *http.Request) *mcp.Server {
		url := request.URL.Path

		switch url {
		case "/read_file":
			return fileOperations

		default:
			return mcp.NewServer(&mcp.Implementation{Name: "unknown_tool", Version: "v1.0.0"}, nil)
		}
	}

	handler := mcp.NewSSEHandler(f)
	log.Fatal(http.ListenAndServe(addr, handler))
}

// =============================================================================
// Tools

type MCPReadFileParams struct {
	Path string `json:"path" jsonschema:"a possible filter to use"`
}

func MCPReadFile(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[MCPReadFileParams]) (*mcp.CallToolResultFor[any], error) {
	dir := "."
	if params.Arguments.Path != "" {
		dir = params.Arguments.Path
	}

	content, err := os.ReadFile(dir)
	if err != nil {
		return nil, err
	}

	v := string(content)
	words := strings.Fields(v)
	if len(words) > 4096 {
		words = words[:4096]
	}

	info := struct {
		Data map[string]any `json:"data"`
	}{
		Data: map[string]any{
			"file_contents": strings.Join(words, " "),
		},
	}

	data, err := json.Marshal(info)
	if err != nil {
		return nil, err
	}

	return &mcp.CallToolResultFor[any]{
		Content: []mcp.Content{&mcp.TextContent{
			Text: string(data),
		}},
	}, nil
}
