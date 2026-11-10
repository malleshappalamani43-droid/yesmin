package com.yesmin.app.ai

import com.yesmin.app.commands.ActionType
import com.yesmin.app.commands.AssistantReply
import com.yesmin.app.core.Msg
import com.yesmin.app.core.YesminException
import com.yesmin.app.data.SecureSettings
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.OkHttpClient
import okhttp3.Request
import okhttp3.RequestBody.Companion.toRequestBody
import org.json.JSONArray
import org.json.JSONException
import org.json.JSONObject
import java.io.IOException

class GeminiClient(private val settings: SecureSettings, private val http: OkHttpClient) {

    /** history: (fromUser, text) pairs, newest last. */
    suspend fun ask(history: List<Pair<Boolean, String>>, memories: List<String>): AssistantReply =
        withContext(Dispatchers.IO) {
            val key = settings.geminiKey
            if (key.isBlank()) throw YesminException(Msg.GEMINI_KEY)

            val contents = JSONArray()
            history.dropWhile { !it.first }.forEach { (fromUser, text) ->
                contents.put(JSONObject()
                    .put("role", if (fromUser) "user" else "model")
                    .put("parts", JSONArray().put(JSONObject().put("text", text))))
            }
            val body = JSONObject()
                .put("systemInstruction", JSONObject().put("parts",
                    JSONArray().put(JSONObject().put("text", systemPrompt(memories)))))
                .put("contents", contents)
                .put("generationConfig", JSONObject().put("responseMimeType", "application/json"))

            val payload = body.toString().toRequestBody("application/json".toMediaType())

            // Try each model in order; on temporary errors (503 etc.) retry with backoff,
            // then fall back to the next model.
            var code = 0
            var raw = ""
            var netError: IOException? = null
            outer@ for (model in MODELS) {
                val req = Request.Builder().url(BASE_URL + model + ":generateContent")
                    .header("x-goog-api-key", key)   // header, not URL, so it never lands in logs
                    .post(payload)
                    .build()
                for (attempt in 0 until ATTEMPTS_PER_MODEL) {
                    try {
                        val r = http.newCall(req).execute().use { it.code to it.body?.string().orEmpty() }
                        code = r.first; raw = r.second; netError = null
                    } catch (e: IOException) {
                        netError = e; code = 0
                    }
                    if (code in 200..299) break@outer
                    // Wrong key / bad request: retrying or switching model won't help.
                    if (code == 401 || code == 403 || (code == 400 && raw.contains("API key", true))) break@outer
                    if (code in 400..499 && code != 429 && code != 408) break@outer
                    if (attempt < ATTEMPTS_PER_MODEL - 1) delay(BACKOFF_MS * (attempt + 1))
                }
            }

            if (code == 0) throw YesminException(Msg.NET, netError)
            if (code !in 200..299) {
                if (code == 401 || code == 403 || (code == 400 && raw.contains("API key", true)))
                    throw YesminException(Msg.GEMINI_BAD_KEY)
                throw YesminException(Msg.GEMINI_FAIL + " (HTTP $code)")
            }
            val text = try {
                JSONObject(raw).optJSONArray("candidates")?.optJSONObject(0)?.optJSONObject("content")
                    ?.optJSONArray("parts")?.optJSONObject(0)?.optString("text")
            } catch (e: JSONException) { null }
            if (text.isNullOrBlank()) throw YesminException(Msg.GEMINI_FAIL)
            parse(text)
        }

    private fun parse(text: String): AssistantReply = try {
        val o = JSONObject(text)
        val action = runCatching { ActionType.valueOf(o.optString("action", "CHAT").uppercase()) }
            .getOrDefault(ActionType.CHAT)
        val speech = o.optString("speech").trim()
        if (speech.isEmpty()) throw YesminException(Msg.GEMINI_FAIL)
        val args = mutableMapOf<String, String>()
        o.optJSONObject("args")?.let { a -> a.keys().forEach { k -> args[k] = a.optString(k) } }
        AssistantReply(action, speech, args)
    } catch (e: JSONException) {
        AssistantReply(ActionType.CHAT, text.trim(), emptyMap())
    }

    private fun systemPrompt(memories: List<String>) = """
You are ${settings.assistantName}, a JARVIS-style voice assistant. The user may call you by this name. Always reply in natural spoken Kannada written in Kannada script (you may use common English words like WhatsApp, YouTube in English letters). Only switch language if the user clearly speaks another language. Keep spoken replies short. No markdown, no emojis.
Respond ONLY with JSON: {"action": "...", "speech": "...", "args": {...}}
Actions:
- CHAT: normal conversation (default).
- OPEN_APP: args {"app_name"} written in English letters as shown on the phone (e.g. YouTube).
- CALL_CONTACT: args {"contact"} written in English letters (e.g. Appa). "speech" must ask for confirmation in Kannada.
- SEND_MESSAGE (WhatsApp): args {"contact" in English letters,"message"}; message is the exact text, in first person as the user, in Kannada. "speech" must ask for confirmation in Kannada.
- PLAY_YOUTUBE: args {"query"} (search text only).
- MEMORY: args {"text"}; only when the user asks you to remember something.
- CAMERA, IMAGE_ANALYSIS, SYSTEM_ACTION: not available yet; use the action if requested and say so briefly.
Known facts about the user: ${if (memories.isEmpty()) "none" else memories.joinToString("; ")}
""".trimIndent()

    companion object {
        private const val BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/"
        /** Tried in order; a busy model falls through to the next one. */
        private val MODELS = listOf("gemini-flash-latest", "gemini-2.5-flash", "gemini-2.5-flash-lite")
        private const val ATTEMPTS_PER_MODEL = 2
        private const val BACKOFF_MS = 1000L
    }
}
