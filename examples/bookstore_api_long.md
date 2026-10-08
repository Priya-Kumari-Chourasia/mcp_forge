# Bookstore API - Complete Developer Guide

This is the long form of the Bookstore guide. It is ordinary prose with no OpenAPI file,
and it is deliberately spread out: the facts about one endpoint are not all in one place.
It exists as a sample input for MCP-Forge's prose path. The API itself is fictional.

## 1. Introduction

The Bookstore platform lets partner shops read our catalogue, show reader reviews and,
for staff accounts, keep stock up to date. The service has been running since 2019 and
serves a few hundred partner shops. Most partners only ever read data; a small number of
staff tools also write to it.

Every request goes to the base URL https://api.bookstore.example/v2 and every response is
JSON encoded as UTF-8. No authentication is needed for the endpoints covered here.

## 2. How this guide is organised

Each section covers one area of the product. Within a section we explain what a call is
for before we explain how to make it. A summary table of every endpoint is at the very end
of the guide. Parameter names are written in backticks. When we say a parameter is
"optional" you can leave it out entirely; when we say it is "in the path" it replaces the
placeholder in curly braces.

## 3. Conventions used by every endpoint

Dates are ISO 8601 strings in UTC. Money is an integer number of minor units, so a price of
1299 means 12.99. Identifiers are positive integers and are never reused. Unknown fields in
a request are ignored rather than rejected, which lets us add fields without breaking older
clients. Field names use snake_case throughout.

Lists are returned inside an object with two keys: `items`, which holds the results, and
`next_page`, which is either the number of the next page or null when there are no more.

## 4. Browsing the catalogue

To list books, call GET /books. Results are paginated, fifty to a page. You may pass the
optional query parameter `page` (an integer, starting at 1) and the optional query
parameter `genre` (a string such as "fiction") to narrow the list. Books come back in the
order they were added, newest first.

To fetch a single book, call GET /books/{book_id}. The `book_id` in the path is the numeric
identifier returned by the listing call. The response includes the title, author, price,
genre and the date the book was added.

## 5. Rate limits

Clients may make 60 requests per minute. Exceeding the limit returns HTTP 429 with a
Retry-After header giving the number of seconds to wait. Limits are applied per IP address,
so several tools running behind one office network share a single allowance. If you
regularly hit the limit, cache catalogue listings for a few minutes; they change slowly.

We do not offer a higher limit for individual partners. Bulk exports are available on
request from the partnerships team and arrive as a file, not through this API.

## 6. Errors

Errors use standard HTTP status codes. The response body always contains an `error` field
with a short machine-readable code and a `message` field written for people.

The codes you are most likely to see are `not_found` when an identifier does not exist,
`invalid_request` when a field has the wrong type, `rate_limited` when you exceed the
allowance described above, and `conflict` when two staff tools change the same book at
once. A 500-range status always means the fault is ours; retry after a short pause.

## 7. Reviews

Readers can leave reviews on any book. GET /books/{book_id}/reviews returns every review
for one book, newest first. The optional query parameter `min_rating` (an integer from 1 to
5) hides reviews below that score. Each review has a rating, the review text, a display
name and the date it was written.

Reviews are moderated before they appear, so a review written a minute ago may not be in
the list yet. Moderation usually takes under an hour.

## 8. Data freshness and caching

Catalogue data is served from a read replica that trails the main database by a few
seconds. If a staff tool adds a book and immediately lists the catalogue, the new book may
be missing from the first response. Wait two seconds, or fetch the book directly by its
identifier, which always reads from the main database.

Responses carry an ETag header. Send it back in If-None-Match and you will receive HTTP 304
with an empty body when nothing has changed, which does not count against your rate limit.

## 9. Client libraries

We publish small client libraries for Python and JavaScript. They wrap the calls in this
guide, retry on 429 and 500-range responses, and handle paging for you. They are optional;
everything they do can be done with any HTTP client.

## 10. Managing stock

Staff tools can change the catalogue.

Adding a title: POST /books creates a new book. Send a JSON body with the fields `title`,
`author` and `price`. The response is the new book, including its identifier.

## 11. Changelog

Version 2 replaced version 1 in 2022. The main differences are that prices became integers
in minor units, list responses gained the `next_page` key, and review moderation was
introduced. Version 1 was switched off at the end of 2023.

Since then we have added the `genre` filter on catalogue listings and the `min_rating`
filter on reviews. Neither change affected existing clients.

## 12. Withdrawing a title

When a book is no longer sold it can be taken out of the catalogue for good. This cannot be
undone, and the identifier is never reused. The call is listed in the endpoint table; it
takes the book's identifier in the path and has no request body.

An optional query flag named `keep_reviews` (true or false) decides whether the book's
reviews are kept in our archive. If you leave it out, reviews are deleted together with the
book. A successful withdrawal returns HTTP 204 with an empty body.

## 13. Frequently asked questions

Can I search by title? Not through this API; use the `genre` filter and search on your side.

Can readers edit a review? No. They can ask moderation to remove it and then write another.

Is there a sandbox? No separate sandbox exists. Read calls are safe to try against the live
service. Do not test the staff calls against live data.

## 14. Endpoint table

| Method | Path |
|---|---|
| GET | /books |
| GET | /books/{book_id} |
| GET | /books/{book_id}/reviews |
| POST | /books |
| DELETE | /books/{book_id} |