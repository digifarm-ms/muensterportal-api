from muenster4you.rag.retrieval import RetrievalResult


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "muenster4you"
    assert data["version"] == "0.1.0"
    assert data["status"] == "ok"


def test_search_without_query(client):
    response = client.get("/search")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Please provide a search query"
    assert data["query"] is None
    assert data["results"] == []


def test_search_with_empty_query(client):
    response = client.get("/search?q=")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Please provide a search query"
    assert data["query"] is None


def _make_retrieval_result(**overrides):
    defaults = {
        "page_id": 1,
        "page_title": "Test Page",
        "content_text": "Some content about Muenster",
        "similarity_score": 0.85,
        "page_len": 100,
        "source": "wiki",
    }
    defaults.update(overrides)
    return RetrievalResult(**defaults)


def test_search_with_results(client, mock_retriever):
    mock_retriever.retrieve.return_value = [
        _make_retrieval_result(),
        _make_retrieval_result(page_id=2, page_title="Second Page", similarity_score=0.7),
    ]
    response = client.get("/search?q=test&top_k=3")
    assert response.status_code == 200
    data = response.json()
    assert data["query"] == "test"
    assert len(data["results"]) == 2
    assert data["results"][0]["page_title"] == "Test Page"
    assert data["results"][1]["page_title"] == "Second Page"
    assert data["message"] is None
    mock_retriever.retrieve.assert_called_once_with("test", top_k=3)


def test_query_endpoint(client, mock_retriever, mock_generator):
    mock_retriever.retrieve.return_value = [_make_retrieval_result()]
    response = client.post("/query", json={"question": "What is Muenster?"})
    assert response.status_code == 200
    data = response.json()
    assert data["question"] == "What is Muenster?"
    assert data["answer"] == "Mock answer"
    assert len(data["sources"]) == 1
    assert data["sources"][0]["page_title"] == "Test Page"
    mock_retriever.retrieve.assert_called_once_with("What is Muenster?", top_k=5)


def test_query_endpoint_custom_params(client, mock_retriever, mock_generator):
    mock_retriever.retrieve.return_value = []
    response = client.post(
        "/query",
        json={"question": "test", "top_k": 3, "temperature": 0.2},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["sources"] == []
    mock_retriever.retrieve.assert_called_once_with("test", top_k=3)
    mock_generator.generate.assert_called_once()
    _, kwargs = mock_generator.generate.call_args
    assert kwargs["temperature"] == 0.2


# --- Chat endpoint tests ---


def test_chat_new_conversation(client, mock_retriever, mock_generator):
    mock_retriever.retrieve.return_value = [_make_retrieval_result()]
    response = client.post("/chat", json={"message": "Was ist Münster?"})
    assert response.status_code == 200
    data = response.json()
    assert data["conversation_id"] is not None
    assert data["answer"] == "Mock chat answer"
    assert data["remaining_followups"] == 3
    assert len(data["history"]) == 2  # user + assistant
    assert data["history"][0]["role"] == "user"
    assert data["history"][0]["content"] == "Was ist Münster?"
    assert data["history"][1]["role"] == "assistant"
    assert len(data["sources"]) == 1
    mock_retriever.retrieve.assert_called_once()
    mock_generator.build_system_message.assert_called_once()
    mock_generator.chat.assert_called_once()


def test_chat_followup(client, mock_retriever, mock_generator):
    mock_retriever.retrieve.return_value = [_make_retrieval_result()]

    # Start conversation
    r1 = client.post("/chat", json={"message": "Was ist Münster?"})
    conversation_id = r1.json()["conversation_id"]

    # Follow-up (no new retrieval)
    mock_retriever.retrieve.reset_mock()
    r2 = client.post(
        "/chat",
        json={"message": "Erzähl mir mehr", "conversation_id": conversation_id},
    )
    assert r2.status_code == 200
    data = r2.json()
    assert data["conversation_id"] == conversation_id
    assert data["remaining_followups"] == 2
    assert len(data["history"]) == 4  # 2 user + 2 assistant
    mock_retriever.retrieve.assert_not_called()


def test_chat_max_followups_exceeded(client, mock_retriever, mock_generator):
    mock_retriever.retrieve.return_value = [_make_retrieval_result()]

    # Start + 3 follow-ups = 4 turns total (the max)
    r = client.post("/chat", json={"message": "Frage 1"})
    cid = r.json()["conversation_id"]
    for i in range(3):
        r = client.post("/chat", json={"message": f"Frage {i+2}", "conversation_id": cid})
    assert r.json()["remaining_followups"] == 0

    # 5th message should be rejected
    r = client.post("/chat", json={"message": "Frage 5", "conversation_id": cid})
    assert r.status_code == 400
    assert "Rückfragen" in r.json()["detail"]


def test_chat_invalid_conversation_id_creates_new(client, mock_retriever, mock_generator):
    mock_retriever.retrieve.return_value = [_make_retrieval_result()]
    response = client.post(
        "/chat",
        json={"message": "Hallo", "conversation_id": "nonexistent-id"},
    )
    assert response.status_code == 200
    data = response.json()
    # Should have created a new session with a different id
    assert data["conversation_id"] != "nonexistent-id"
    assert data["remaining_followups"] == 3
    mock_retriever.retrieve.assert_called_once()  # retrieval ran (new session)
