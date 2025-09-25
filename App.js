import React, { useState, useRef, useEffect } from "react";
import { Button } from './components/ui/button';
import "./App.css";

const VoiceAssistant = () => {
  const [text, setText] = useState("");
  const [spokenText, setSpokenText] = useState("");
  const [responseText, setResponseText] = useState("");
  const [faqResponse, setFaqResponse] = useState("");
  const [audioUrl, setAudioUrl] = useState("");
  const [escalationMessage, setEscalationMessage] = useState("");
  const [isRecording, setIsRecording] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const recognitionRef = useRef(null);
  const audioRef = useRef(null); 

  const handlePredict = async (inputText) => {
    if (isLoading) return;
    setIsLoading(true);
    setError("");

    try {
      const response = await fetch("http://localhost:5000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text: inputText, user_id: 1 }),
      });

      if (!response.ok) {
        throw new Error(`Backend error: ${response.statusText}`);
      }

      const data = await response.json();

      if (data.escalated) {
        setEscalationMessage(data.message);
        setResponseText("");
        setFaqResponse("");
        setAudioUrl("");
      } else {
        setEscalationMessage("");
        setResponseText(data.response_text || "");
        setFaqResponse("");
        if (data.audio_file) {
          const url = `http://localhost:5000${data.audio_file}`;
          setAudioUrl(url);
        }
      }
    } catch (error) {
      console.error("Error in prediction flow:", error);
      setError("Failed to get a response from the server. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  const handleSubmit = () => {
    if (!text.trim()) {
      setError("Please enter a question before submitting.");
      return;
    }
    handlePredict(text);
  };

  const handleStartRecording = async () => {
    setError("");
    setIsRecording(true);

    const SpeechRecognition =
      window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
      try {
        await navigator.mediaDevices.getUserMedia({ audio: true });

        const recognition = new SpeechRecognition();
        recognitionRef.current = recognition;
        recognition.continuous = false;
        recognition.interimResults = false;
        recognition.lang = "en-US";

        recognition.onresult = (event) => {
          const transcript = event.results[0][0].transcript;
          setSpokenText(transcript);
          handlePredict(transcript);
          setIsRecording(false);
        };

        recognition.onerror = (event) => {
          console.error("Speech recognition error:", event.error);
          setIsRecording(false);
          if (event.error === "not-allowed") {
            setError("Microphone access denied. Please allow microphone access in your browser settings.");
          } else {
            setError(`Speech recognition error: ${event.error}`);
          }
        };

        recognition.onend = () => {
          setIsRecording(false);
        };

        recognition.start();
      } catch (err) {
        console.error("Microphone permission denied or unavailable:", err);
        setError("Microphone access denied. Please allow microphone access in your browser settings.");
        setIsRecording(false);
        handleMediaRecorderFallback();
      }
    } else {
      console.warn("SpeechRecognition not supported, using fallback.");
      setError("Speech recognition not supported in this browser. Using fallback method.");
      handleMediaRecorderFallback();
    }
  };

  const handleMediaRecorderFallback = () => {
    navigator.mediaDevices.getUserMedia({ audio: true }).then((stream) => {
      const mediaRecorder = new MediaRecorder(stream);
      const audioChunks = [];

      mediaRecorder.ondataavailable = (event) => {
        audioChunks.push(event.data);
      };

      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: "audio/wav" });
        const formData = new FormData();
        formData.append("audio", audioBlob);

        try {
          const sttResponse = await fetch("http://localhost:5000/stt", {
            method: "POST",
            body: formData,
          });

          if (!sttResponse.ok) {
            throw new Error(`STT endpoint error: ${sttResponse.statusText}`);
          }

          const sttData = await sttResponse.json();
          if (sttData.error) {
            throw new Error(sttData.error);
          }

          const transcribedText = sttData.text;
          setSpokenText(transcribedText);
          handlePredict(transcribedText);
        } catch (error) {
          console.error("Error during STT:", error);
          setError(`Failed to transcribe audio: ${error.message}`);
        } finally {
          setIsRecording(false);
        }
      };

      mediaRecorder.start();
      setTimeout(() => {
        mediaRecorder.stop();
      }, 5000);
    }).catch((err) => {
      console.error("MediaRecorder fallback failed:", err);
      setError("Microphone access denied or unavailable. Please allow microphone access.");
      setIsRecording(false);
    });
  };

  // Effect to handle audio playback when audioUrl changes
  useEffect(() => {
    if (audioUrl && audioRef.current) {
      audioRef.current.load(); // Reload the audio element with the new source
      const playPromise = audioRef.current.play();

      if (playPromise !== undefined) {
        playPromise.catch((err) => {
          console.error("Autoplay failed:", err);
          setError("Autoplay blocked by browser. Please click the play button to hear the response.");
        });
      }
    }
  }, [audioUrl]);

  return (
    <div className="voice-assistant-container">
      <img src="/logo.png" alt="Bank Logo" className="logo" />
      <h1 className="title">NLP CUSTOMER SUPPORT IN BANKING</h1>

      {error && (
        <div className="mt-4">
          <p className="text-red-600 font-semibold">{error}</p>
        </div>
      )}

      <label className="block mb-1 font-medium">Type Your Question</label>
      <textarea
        className="textarea"
        rows="4"
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder="Type your question here..."
        disabled={isLoading}
      />

      <label className="block mt-4 mb-1 font-medium">Transcribed from Voice</label>
      <textarea
        className="textarea"
        rows="4"
        value={spokenText}
        readOnly
        placeholder="Your spoken question will appear here..."
      />

      <div className="button-row mt-4">
        <Button onClick={handleSubmit} disabled={isLoading}>
          {isLoading ? "Processing..." : "📤 Submit"}
        </Button>
        {!isRecording ? (
          <Button onClick={handleStartRecording} disabled={isLoading}>
            🎙️ Voice Input
          </Button>
        ) : (
          <Button disabled>⏳ Listening...</Button>
        )}
      </div>

      {escalationMessage && (
        <div className="mt-4">
          <p className="font-semibold mb-2 text-red-600">🚨 Escalation Notice:</p>
          <p className="response-box bg-yellow-100 border border-red-400">
            {escalationMessage}
          </p>
        </div>
      )}

      {!escalationMessage && responseText && (
        <div className="mt-4">
          <p className="font-semibold mb-2">💬 Response:</p>
          <p className="response-box">{responseText}</p>
          {audioUrl && (
            <div className="mt-2">
              <audio
                ref={audioRef}
                controls
                autoPlay 
                onError={() => setError("Failed to load audio file.")}
              >
                <source src={audioUrl} type="audio/mpeg" />
                Your browser does not support the audio element.
              </audio>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default VoiceAssistant;

