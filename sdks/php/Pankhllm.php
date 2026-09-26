<?php
// pankhllm client for PHP 8+. Uses ext-curl and ext-json only.
declare(strict_types=1);

namespace Pankhllm;

final class PankhException extends \RuntimeException
{
    public function __construct(public readonly int $status, public readonly string $kind, string $message)
    {
        parent::__construct("$status $kind: $message");
    }
}

final class Client
{
    public function __construct(private string $baseUrl = 'http://localhost:4000', private array $headers = [])
    {
        $this->baseUrl = rtrim($baseUrl, '/');
    }

    /** @param array{messages?:array,context?:array,tags?:array,prompt?:string,tier?:string,intent?:string,model?:string,max_tokens?:int,max_cost_usd?:float,max_latency_ms?:int,temperature?:float} $o */
    private function body(?string $question, array $o, bool $stream): array
    {
        $ext = [];
        foreach (['context', 'tags', 'prompt', 'tier', 'intent', 'max_cost_usd', 'max_latency_ms'] as $k) {
            if (isset($o[$k]) && $o[$k] !== [] && $o[$k] !== null) $ext[$k] = $o[$k];
        }
        $b = [
            'model' => $o['model'] ?? 'auto',
            'messages' => $o['messages'] ?? [['role' => 'user', 'content' => $question ?? '']],
            'stream' => $stream,
            'pankhllm' => (object) $ext,
        ];
        if (isset($o['max_tokens'])) $b['max_tokens'] = $o['max_tokens'];
        if (isset($o['temperature'])) $b['temperature'] = $o['temperature'];
        return $b;
    }

    private function post(string $path, array $body, ?callable $onLine = null): string
    {
        $ch = curl_init($this->baseUrl . $path);
        $headers = ['Content-Type: application/json'];
        foreach ($this->headers as $k => $v) $headers[] = "$k: $v";
        curl_setopt_array($ch, [
            CURLOPT_POST => true,
            CURLOPT_POSTFIELDS => json_encode($body, JSON_THROW_ON_ERROR),
            CURLOPT_HTTPHEADER => $headers,
            CURLOPT_TIMEOUT => 120,
        ]);
        $out = '';
        if ($onLine) {
            $buf = '';
            curl_setopt($ch, CURLOPT_WRITEFUNCTION, function ($ch, $chunk) use (&$buf, $onLine) {
                $buf .= $chunk;
                while (($pos = strpos($buf, "\n")) !== false) {
                    $line = substr($buf, 0, $pos);
                    $buf = substr($buf, $pos + 1);
                    $onLine(rtrim($line, "\r"));
                }
                return strlen($chunk);
            });
        } else {
            curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
        }
        $res = curl_exec($ch);
        if ($res === false) throw new PankhException(0, 'transport', curl_error($ch));
        $status = (int) curl_getinfo($ch, CURLINFO_RESPONSE_CODE);
        curl_close($ch);
        if (!$onLine) $out = (string) $res;
        if ($status >= 400) {
            $e = json_decode($out, true)['error'] ?? [];
            throw new PankhException($status, $e['type'] ?? 'error', $e['message'] ?? $out);
        }
        return $out;
    }

    /** Route and answer. Returns the decoded OpenAI-shaped response; routing info is under ['pankhllm']. */
    public function ask(?string $question, array $o = []): array
    {
        return json_decode($this->post('/v1/chat/completions', $this->body($question, $o, false)), true, 512, JSON_THROW_ON_ERROR);
    }

    /** Streams text deltas to $onDelta; $onMeta (optional) receives the routing metadata first. */
    public function stream(?string $question, array $o, callable $onDelta, ?callable $onMeta = null): void
    {
        $metaSent = false;
        $this->post('/v1/chat/completions', $this->body($question, $o, true), function (string $line) use (&$metaSent, $onDelta, $onMeta) {
            if (!str_starts_with($line, 'data:')) return;
            $data = trim(substr($line, 5));
            if ($data === '[DONE]') return;
            $ev = json_decode($data, true);
            if (!$metaSent && isset($ev['pankhllm']) && $onMeta) { $metaSent = true; $onMeta($ev['pankhllm']); }
            foreach ($ev['choices'] ?? [] as $ch) {
                if (($ch['finish_reason'] ?? null) === 'error') throw new PankhException(502, 'upstream_error', json_encode($ev['pankhllm'] ?? []));
                $d = $ch['delta']['content'] ?? '';
                if ($d !== '') $onDelta($d);
            }
        });
    }

    /** Dry run: routing decision only. */
    public function route(?string $question, array $o = []): array
    {
        return json_decode($this->post('/v1/route', $this->body($question, $o, false)), true, 512, JSON_THROW_ON_ERROR);
    }
}
