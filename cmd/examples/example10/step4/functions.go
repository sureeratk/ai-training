package main

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/ardanlabs/ai-training/foundation/client"
)

// WE WILL ADD SUPPORT FOR STRUCTURED TOOL RESPONSES.

// toolSuccessResponse returns a successful structured tool response.
func toolSuccessResponse(toolID string, toolName string, keyValues ...any) client.D {
	data := make(map[string]any)
	for i := 0; i < len(keyValues); i = i + 2 {
		data[keyValues[i].(string)] = keyValues[i+1]
	}

	return toolResponse(toolID, toolName, data, "SUCCESS")
}

// toolErrorResponse returns a failed structured tool response.
func toolErrorResponse(toolID string, toolName string, err error) client.D {
	data := map[string]any{"error": err.Error()}

	return toolResponse(toolID, toolName, data, "FAILED")
}

// toolResponse creates a structured tool response.
func toolResponse(toolID string, toolName string, data map[string]any, status string) client.D {
	info := struct {
		Status string         `json:"status"`
		Data   map[string]any `json:"data"`
	}{
		Status: status,
		Data:   data,
	}

	content, err := json.Marshal(info)
	if err != nil {
		return client.D{
			"role":         "tool",
			"tool_call_id": toolID,
			"tool_name":    toolName,
			"content":      `{"status": "FAILED", "data": "error marshaling tool response"}`,
		}
	}

	return client.D{
		"role":         "tool",
		"tool_call_id": toolID,
		"tool_name":    toolName,
		"content":      string(content),
	}
}

// =============================================================================

// WE WILL DEFINE A TYPE FOR THE TOOL.

// GetWeather represents a tool that can be used to get the current weather.
type GetWeather struct {
	name string
}

// RegisterGetWeather creates a new instance of the GetWeather tool and loads it
// into the provided tools map.
func RegisterGetWeather(tools map[string]Tool) client.D {
	gw := GetWeather{
		name: "tool_get_weather",
	}
	tools[gw.name] = &gw

	return gw.toolDocument()
}

// toolDocument defines the metadata for the tool that is provied to the model.
func (gw *GetWeather) toolDocument() client.D {
	return client.D{
		"type": "function",
		"function": client.D{
			"name":        gw.name,
			"description": "Get the current weather for a location",
			"parameters": client.D{
				"type": "object",
				"properties": client.D{
					"location": client.D{
						"type":        "string",
						"description": "The location to get the weather for, e.g. San Francisco, CA",
					},
				},
				"required": []string{"location"},
			},
		},
	}
}

// Call is the function that is called by the agent to get the weather when the
// model requests the tool with the specified parameters.
func (gw *GetWeather) Call(ctx context.Context, toolCall client.ToolCall) (resp client.D) {
	defer func() {
		if r := recover(); r != nil {
			resp = toolErrorResponse(toolCall.ID, gw.name, fmt.Errorf("%s", r))
		}
	}()

	// We are going to hardcode a result for now so we can test the tool. The
	// data weather will be returned as structured data using JSON which is
	// easier for the model to interpret.

	location := toolCall.Function.Arguments["location"].(string)

	data := map[string]any{
		"temperature": 28,
		"humidity":    80,
		"wind_speed":  10,
		"description": fmt.Sprintln("The weather in", location, "is hot and humid"),
	}

	content, err := json.Marshal(data)
	if err != nil {
		return toolErrorResponse(toolCall.ID, gw.name, err)
	}

	return toolSuccessResponse(toolCall.ID, gw.name, "weather", string(content))
}
