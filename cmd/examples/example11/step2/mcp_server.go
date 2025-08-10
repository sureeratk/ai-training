package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/exec"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

// mcpListenAndServe starts the MCP server for all the tooling we support.
func mcpListenAndServe(host string) {
	fileOperations := mcp.NewServer(&mcp.Implementation{Name: "file_operations", Version: "v1.0.0"}, nil)

	RegisterReadFileTool(fileOperations)
	RegisterShellCommandTool(fileOperations)

	// -------------------------------------------------------------------------

	fmt.Printf("\nServer: MCP servers serving at %s\n", host)

	// -------------------------------------------------------------------------

	f := func(request *http.Request) *mcp.Server {
		url := request.URL.Path

		switch url {
		case "/read_file", "/search_file", "/create_file", "/go_code_editor":
			return fileOperations

		default:
			return mcp.NewServer(&mcp.Implementation{Name: "unknown_tool", Version: "v1.0.0"}, nil)
		}
	}

	handler := mcp.NewSSEHandler(f)
	fmt.Println(http.ListenAndServe(host, handler))
}

// =============================================================================

// RegisterReadFileTool registers the read_file tool with the given MCP server.
func RegisterReadFileTool(mcpServer *mcp.Server) {
	const toolName = "read_file"
	const tooDescription = "Read the contents of a given file path or search for files containing a pattern. When searching file contents, returns line numbers where the pattern is found."

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, ReadFileTool)
}

// ReadFileToolParams represents the parameters for this tool call.
type ReadFileToolParams struct {
	Path string `json:"path" jsonschema:"a possible filter to use"`
}

// ReadFileTool reads the contents of a given file path. It will limit the
// result to 4096 words.
func ReadFileTool(ctx context.Context, cc *mcp.ServerSession, params *mcp.CallToolParamsFor[ReadFileToolParams]) (*mcp.CallToolResultFor[any], error) {
	const maxWords = 4096

	dir := "."
	if params.Arguments.Path != "" {
		dir = params.Arguments.Path
	}

	content, err := os.ReadFile(dir)
	if err != nil {
		return nil, err
	}

	info := struct {
		Data map[string]any `json:"data"`
	}{
		Data: map[string]any{
			"file_contents": string(content),
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

// =============================================================================

// RegisterShellCommandTool registers the shell_command tool with the given MCP server.
func RegisterShellCommandTool(mcpServer *mcp.Server) {
	const toolName = "shell_command"
	const tooDescription = "Execute a shell command with parameters and return the output."

	mcp.AddTool(mcpServer, &mcp.Tool{Name: toolName, Description: tooDescription}, ShellCommand)
}

type ShellCommandParams struct {
	Command []string `json:"command" jsonschema:"the command and arguments to execute"`
}

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
