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
