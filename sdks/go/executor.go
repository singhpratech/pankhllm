package pankhllm

// Planner-lane executor for Go: a net/http handler.
//
//	http.Handle("/execute", pankhllm.Executor(map[string]pankhllm.Op{
//	    "metric_by_period": func(ctx context.Context, p map[string]any, pc pankhllm.PlanContext) (any, error) {
//	        return map[string]any{"value": kpi(ctx, p["metric"].(string), pc.User)}, nil
//	    },
//	}, os.Getenv("PANKH_EXECUTOR_KEY")))

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

// PlanContext carries who asked, for row-level scope.
type PlanContext struct {
	Question string `json:"question"`
	User     string `json:"user"`
	Tenant   string `json:"tenant"`
	TraceID  string `json:"trace_id"`
}

// Op runs one validated plan. Return a string (the answer), a slice (rows) or any JSON value (data).
type Op func(ctx context.Context, params map[string]any, pc PlanContext) (any, error)

// HandlePlan runs a plan and never panics: failures become {"ok": false} so the router falls through.
func HandlePlan(ctx context.Context, payload map[string]any, ops map[string]Op) (out map[string]any) {
	defer func() {
		if r := recover(); r != nil {
			out = map[string]any{"ok": false, "error": fmt.Sprint(r)}
		}
	}()
	name, _ := payload["op"].(string)
	fn, ok := ops[name]
	if !ok {
		return map[string]any{"ok": false, "error": "unknown operation " + name}
	}
	params, _ := payload["params"].(map[string]any)
	var pc PlanContext
	pc.Question, _ = payload["question"].(string)
	pc.User, _ = payload["user"].(string)
	pc.Tenant, _ = payload["tenant"].(string)
	pc.TraceID, _ = payload["trace_id"].(string)
	res, err := fn(ctx, params, pc)
	if err != nil {
		return map[string]any{"ok": false, "error": err.Error()}
	}
	switch v := res.(type) {
	case string:
		return map[string]any{"ok": true, "answer": v}
	case []any, []map[string]any:
		return map[string]any{"ok": true, "data": map[string]any{"rows": v}}
	default:
		return map[string]any{"ok": true, "data": v}
	}
}

// Executor returns the HTTP handler. apiKey "" disables the bearer check.
func Executor(ops map[string]Op, apiKey string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if r.Method != http.MethodPost {
			w.WriteHeader(http.StatusMethodNotAllowed)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "POST only"})
			return
		}
		if apiKey != "" && r.Header.Get("Authorization") != "Bearer "+apiKey {
			w.WriteHeader(http.StatusUnauthorized)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "unauthorized"})
			return
		}
		var payload map[string]any
		if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<20)).Decode(&payload); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			_ = json.NewEncoder(w).Encode(map[string]any{"ok": false, "error": "bad json"})
			return
		}
		_ = json.NewEncoder(w).Encode(HandlePlan(r.Context(), payload, ops))
	})
}
