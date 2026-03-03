# Beaver's Choice Paper Co: Multi-Agent Order System

# Agent Workflow Diagram

## Architecture Overview

The system follows a **hierarchical orchestration pattern** where a central orchestrator delegates work sequentially to three specialist agents, each equipped with dedicated tools.

Customer order requests flow through a structured pipeline: **inventory validation → pricing/quoting → transaction execution**.

## Workflow (Sequential)

**Customer Request → Orchestrator**

The **Order Request Orchestrator** receives raw customer text (e.g., "200 sheets of glossy paper, 100 cardstock, deliver by April 15"). It uses the LLM to parse the request, extracting structured entities: item names (as stated by the customer), quantities, request date, and delivery date. These are passed to its single tool, **_process_order_request_**, which coordinates the three-stage pipeline.

**Stage 1 - Inventory Agent** (Match & Check)

The **Inventory Agent** validates that the requested items exist and are available. It uses two tools sequentially:

- **match_items_to_catalog** - Maps fuzzy customer item names (e.g., "heavy cardstock", "A4 glossy paper") to exact catalog names using keyword matching. Items that cannot be matched are flagged as NOT_IN_CATALOG. Then check the availability of the ordered items by **reading** the **inventory** table.
- **check_stock_level** - For each matched item, verifies current stock against the requested quantity and computes reorder needs based on minimum stock thresholds. The inventory roles here is not limited to check the stocks but also to recommend the required stock to order after completing this order and with the regard to the minimum stock level to always keep for each item. The agent **reads** the **inventory** and transactions tables to calculate net stock levels.

If any item is not in the catalog or not currently stocked, the agent returns an error and the pipeline stops immediately.

**Stage 2 - Quoting Agent** (Pricing & Discount)

The **Quoting Agent** determines pricing strategy and generates a quote. It uses one tool:

- **get_quote** - Computes the base price from catalog unit prices, then applies a discount strategy derived from historical quotes. The a default discount strategy is based on three order volume categories and discount rates: small (5%), medium (10%) and large (15%). The volume thresholds are calculated based on the previous order metadata. The agent will try to match order similar to the current ones, first by looking at order with the same items and if not, with the same volume category. In the second case, the method will look ate the order with similar size and maximum number of common items as best match.

The agent **reads** the _quotes_ and _quote_requests_ tables (Quote History) to find similar past orders by keyword and order size. The best-matching historical quote determines the discount type (percentage or rounding) and rate. The tool calculates per-item discounted prices (needed later for the transactions) and returns the full quote breakdown.

**Stage 3 - Transactions Agent (Execution)**

The **Transactions** Agent finalizes the order by executing financial transactions. This is the only agent who is able to write in the tables

It uses one tool:

- **complete_transactions** - First, determines if stock reordering is needed (provided by the inventory agent) and calculates supplier delivery dates. If the earliest possible delivery exceeds the customer's requested date, the order is rejected. Otherwise, it verifies cash sufficiency, then **writes** stock order transactions (at supplier cost with a 30% margin, an assumption taken so that the catalog prices are higher than the supplier prices, the margin) and sales transactions (at discounted customer price) to the transactions table. Returns the updated cash balance and inventory state.

This agent is intentionally limited to max_steps=1 to prevent duplicate transactions from corrupting the database.

**Response → Customer**

The orchestrator assembles the final response: a customer-facing email-style message with item breakdown, unit prices, applied discount, and total - or an error explanation if any stage failed.

If successful the final message is written in an email style providing one example to the Orchestrator agent prompt.

# Results Discussion

**Evaluation Results**

Out of 20 requests processed (see _test_results.csv_), 4 were successfully fulfilled (R1, R3, R4, R10), resulting in 3 distinct cash balance changes totaling +\$336.69 in net revenue. The remaining 16 requests were rejected for the following reasons: 12 due to items not being available in inventory, 2 because the requested items (balloons, tickets) are not paper products and don't exist in the catalog, and 2 because the supplier could not meet the customer's delivery deadline.

**Limitation : High Rejection Rate**

The primary driver is the inventory coverage constraint: only 40% of the catalog items are stocked in inventory. Since orders are rejected entirely if even one item is unavailable, this creates a compounding effect - an order with three items where one is unstocked means all three are rejected. Items like Matte paper, Construction paper, Poster paper, and Paper napkins are all valid catalog products but simply weren't selected in the initial 40% inventory seed. This "all-or-nothing" policy, while safe from a customer promise perspective, leads to a 60% rejection rate from inventory alone.

**Further improvements**

Three practical enhancements would significantly reduce rejections:

- Implementing partial fulfillment would allow the system to process the available items and inform the customer about the unavailable ones, letting them decide whether to proceed.
- Enabling supplier procurement for out-of-inventory items would allow the system to order from the supplier when the delivery timeline permits, converting many of the 12 inventory rejections into fulfilled orders with slightly longer lead times.