# Bookstore API - Developer Guide

Welcome to the Bookstore API. This guide is written as ordinary prose, the way many
internal APIs are documented, with no OpenAPI file. It exists as a sample input for
MCP-Forge's prose path. The API itself is fictional.

## Getting started

Every request goes to the base URL https://api.bookstore.example/v2 and returns JSON.
No authentication is needed for the endpoints in this guide.

## Browsing the catalogue

To list books, call GET /books. Results are paginated. You may pass the optional query
parameter `page` (an integer, starting at 1) and the optional query parameter `genre`
(a string such as "fiction") to narrow the list.

To fetch a single book, call GET /books/{book_id}. The `book_id` in the path is the
numeric identifier returned by the listing call.

## Reviews

Readers can leave reviews. GET /books/{book_id}/reviews returns every review for one
book. The optional query parameter `min_rating` (an integer from 1 to 5) hides reviews
below that score.

## Managing stock

Staff tools can change the catalogue.

Adding a title: POST /books creates a new book. Send a JSON body with the fields
`title`, `author` and `price`.

Withdrawing a title: when a book is no longer sold it can be removed from the catalogue
for good. The request for this is described in the endpoint table at the end of this
guide; it takes the book's identifier in the path. An optional query flag named
`keep_reviews` (true or false) decides whether the book's reviews are kept.

## Rate limits

Clients may make 60 requests per minute. Exceeding the limit returns HTTP 429 with a
Retry-After header. Limits are applied per IP address.

## Errors

Errors use standard HTTP status codes. The response body always contains an `error`
field with a short machine-readable code and a `message` field for people.

## Endpoint table

| Method | Path |
|---|---|
| GET | /books |
| GET | /books/{book_id} |
| GET | /books/{book_id}/reviews |
| POST | /books |
| DELETE | /books/{book_id} |
