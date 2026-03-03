import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union, Any
from sqlalchemy import create_engine, Engine
import re
import json

from smolagents import (
    ToolCallingAgent,
    OpenAIServerModel,
    tool,
)

# Load your OpenAI API key
dotenv.load_dotenv(dotenv_path='.env')
openai_api_key = os.getenv('OPENAI_API_KEY')

model = OpenAIServerModel(
    model_id='gpt-4o-mini',
    api_key=openai_api_key,
)

# Ensure working directory is the project folder (where CSV files live)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# I take the assumption that the listed catalog price are not the supplier price 
# (otheriwse the company will loose money taking into account the discount effect)
# I will apply a SUPPLY_MARGIN_PCT of 30% to all catalog prices to get the supplier/inventory price.
SUPPLY_MARGIN_PCT = 0.3

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date AND item_name IS NOT NULL
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################


# Set up and load your env parameters and instantiate your model.

@tool
def match_items_to_catalog(customer_items: List[Dict[str, str]]) -> str:
    """Match customer-requested item names to exact catalog names.
    
    Args:
        customer_items: A list of dictionaries containing the order details of each ordered item format: [{"item_name":<item_1>, "quantity":<number of items>}, {"item_name":<item_2>, "quatity":<number of items>},...]
                
    
    Returns:
        A list of dictionaries containing the matched items with the format containing the ordered items with format: [{"item_name":<item_1>, "quantity":<number of items>, "catalog_name":<catalog_name_or_NOT_IN_CATALOG>}, {"item_name":<item_2>, "quantity":<number of items>, "catalog_name":<catalog_name_or_NOT_IN_CATALOG>},...]
    """
    # Keyword → catalog name mapping (ordered: most specific first)
    KEYWORD_MAP = [
        # Products (check first — exact noun matches)
        (["plates"],                    "Paper plates"),
        (["napkins", "table napkins"],  "Paper napkins"),
        (["paper cups", "cups biodegradable"], "Paper cups"),
        (["disposable cups"],           "Disposable cups"),
        (["table covers"],              "Table covers"),
        (["envelopes"],                 "Envelopes"),
        (["sticky notes"],              "Sticky notes"),
        (["notepads"],                  "Notepads"),
        (["invitation"],                "Invitation cards"),
        (["flyers", "flyer"],           "Flyers"),
        (["streamers"],                 "Party streamers"),
        (["washi tape", "decorative tape"], "Decorative adhesive tape (washi tape)"),
        (["party bags"],                "Paper party bags"),
        (["name tags", "lanyards"],     "Name tags with lanyards"),
        (["presentation folders"],      "Presentation folders"),
        
        # Large format
        (["24x36", "24\" x 36\"", "24 x 36", "poster board"], "Large poster paper (24x36 inches)"),
        (["36-inch", "banner roll"],    "Rolls of banner paper (36-inch width)"),
        
        # Specialty
        (["100 lb", "cover stock"],     "100 lb cover stock"),
        (["80 lb", "text paper"],       "80 lb text paper"),
        (["250 gsm"],                   "250 gsm cardstock"),
        (["220 gsm"],                   "220 gsm poster paper"),
        
        # Paper types (specific features first, generic last)
        (["glossy"],                    "Glossy paper"),
        (["matte"],                     "Matte paper"),
        (["cardstock"],                 "Cardstock"),
        (["recycled"],                  "Recycled paper"),
        (["eco-friendly", "eco friendly"], "Eco-friendly paper"),
        (["construction"],              "Construction paper"),
        (["kraft"],                     "Kraft paper"),
        (["glitter"],                   "Glitter paper"),
        (["decorative"],                "Decorative paper"),
        (["wrapping"],                  "Wrapping paper"),
        (["letterhead"],                "Letterhead paper"),
        (["legal"],                     "Legal-size paper"),
        (["crepe"],                     "Crepe paper"),
        (["photo"],                     "Photo paper"),
        (["uncoated"],                  "Uncoated paper"),
        (["butcher"],                   "Butcher paper"),
        (["heavyweight", "heavy weight"], "Heavyweight paper"),
        (["bright-colored", "bright colored"], "Bright-colored paper"),
        (["patterned"],                 "Patterned paper"),
        (["colored", "color"],          "Colored paper"),
        (["poster"],                    "Poster paper"),
        (["banner"],                    "Banner paper"),
        
        # Generic paper (LAST — only matches if nothing above matched)
        (["letter-sized", "8.5x11", "8.5\"x11\""], "Letter-sized paper"),
        (["a4"],                        "A4 paper"),
        (["printer", "printing", "copy paper", "print paper"], "Standard copy paper"),
    ]

    results=[]
    for item in customer_items:
        try:
            item_name = item["item_name"]
            quantity = int(item["quantity"])
            matched = None
            for keywords, catalog_name in KEYWORD_MAP:
                if any(kw in item_name.lower() for kw in keywords):
                    matched = catalog_name
                    break
            
            results.append({
                "item_name": item_name,
                "quantity": quantity,
                "catalog_name": matched or 'NOT_IN_CATALOG'
            })
        except Exception as e:
            return {"status": "stop", "message": f"Error parsing item: {item}"}


    return {"status": "success", "message": ",".join(
        f"{r['item_name']}:{r['quantity']}:{r['catalog_name']}" for r in results
    )}

# Tools for inventory agent
"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""

def get_inventory_item_names() -> List[str]:
    """Return all item names in the inventory table or catalog as a list."""
    df = pd.read_sql("SELECT item_name FROM inventory", db_engine)
    return df["item_name"].tolist()

def get_inventory_item_min_stock_level(item_name: str) -> int:
    """Return the minimum stock level of an item."""
    df = pd.read_sql("SELECT min_stock_level FROM inventory WHERE LOWER(item_name) = LOWER(:item_name)", db_engine, params={"item_name": item_name})
    return int(df["min_stock_level"].iloc[0])

def get_inventory_item_unit_price(item_name: str) -> int:
    """Return the unit price of an item as per the catalog."""
    df = pd.read_sql("SELECT unit_price FROM inventory WHERE LOWER(item_name) = LOWER(:item_name)", db_engine, params={"item_name": item_name})
    return float(df["unit_price"].iloc[0])

class InventoryAgent(ToolCallingAgent):
    """Agent that manages inventory."""

    def __init__(self, model: OpenAIServerModel):
        self.model = model
        self.inventor_items = get_inventory_item_names()
        
        @tool
        def check_stock_level(ordered_items: List[Dict[str, str]], request_date: str, delivery_date: str) -> List[Dict[str, str]]:
            """Check if all ordered items are in the catalog and if the stock level is sufficient.
            Give recommendation to refill the stock if needed with the recommended quantity.
                Args:
                    ordered_items: A list of dictionaries containing the ordered items with format: [{"item_name":<item_1>, "quantity":<number of items>, "catalog_name":<catalog_name_or_NOT_IN_CATALOG>}, {"item_name":<item_2>, "quantity":<number of items>, "catalog_name":<catalog_name_or_NOT_IN_CATALOG>},...]
                    request_date: The request date in ISO format (YYYY-MM-DD)
                    delivery_date: The delivery date in ISO format (YYYY-MM-DD)
                Returns:
                - A list of text strings for the matched items with format: 
                 ["{item_name}:{quantity}:{in_stock}:{reorder_quantity}", ...], where in_stock is true or false and reorder_quantity is the quantity to reorder if needed to keep the minimum stock level.
 
                - An error message if the processing did not succeed or stop.
            """

            items_in_catalog = {}
            # Check if all items are in the catalog
            not_in_catalog_items = []
            unavailable_items = []
            for item in ordered_items:
                item_name = item["item_name"]
                quantity = 0
                try:
                    quantity = int(item["quantity"])
                    catalog_item_name = item["catalog_name"]
                    
                    if catalog_item_name.lower() == "not_in_catalog":
                        not_in_catalog_items.append(f"Item '{item_name}' is not in the catalog")
                    
                    elif catalog_item_name not in self.inventor_items:
                        unavailable_items.append(f"Item '{item_name}' is not available for the moment.")
                    else:
                        items_in_catalog [catalog_item_name] = quantity

                except Exception as e:
                    print(e)
                    return {"status": "stop", "message": f"Error: Item {item_name} is not in the catalog"}

            if len(unavailable_items) > 0:
                err_msg = "Sorry, we are unable to process your request due to the following issues:\n"
                if len(not_in_catalog_items) > 0:
                    err_msg += f"- The following items: {', '.join(not_in_catalog_items)}\n"
                if len(unavailable_items) > 0:
                    err_msg += f"- The following items: {', '.join(unavailable_items)} \n"
                return {"status": "stop", "message": f"Error: {err_msg}"}

            items_stock_status = []
            all_inventory = get_all_inventory(request_date)

            # Check the stock level of each ordered item    
            for item_name, quantity in items_in_catalog.items():                
                stock_level = all_inventory.get(item_name, 0)
                
                min_stock_level = get_inventory_item_min_stock_level(item_name)
                
                in_stock = stock_level >= quantity
                stock_after_order = stock_level - quantity
                reorder_quantity = int(max(0, min_stock_level - stock_after_order))

                items_stock_status.append(f"{item_name}:{quantity}:{in_stock}:{reorder_quantity}")
            
          
            return {"status": "success", "message": f"items_stock_status: {','.join(items_stock_status)}"}

        super().__init__(
            tools=[match_items_to_catalog, check_stock_level],
            model=model,
            name="inventory_agent",
            description="Agent that manages inventory.",
        )


# Tools for quoting agent
class QuotingAgent(ToolCallingAgent):
    """Agent that provide quotes efficiently, applying bulk discounts strategically to encourage sales."""

    def __init__(self, model: OpenAIServerModel):
        self.model = model

        # Last quotation result
        self.quote_result = {}

        def get_quote_strategy(order_items: List[Dict[str, str]], base_price: float) -> dict:
            """Determine pricing strategy from historical quotes.
            
            Steps:
            1. Classify order size from base_price
            2. Search historical quotes using product keywords and order size
            3. Pick best match (most product overlap, else most recent)
            4. Extract discount strategy from matched explanation: round or percentage bulk discount
            5. Fallback to default if extraction fails

            Args:
                order_items: List of dictionaries containing the ordered items with format: [{"item_name":<item_1>, "quantity":<number of items>}, {"item_name":<item_2>, "quantity":<number of items>},...]
                base_price: The base price of the ordered items

            Returns:
                A dictionary containing the discount strategy:
                    - type: The type of discount applied: "standard", "rounding" or "percentage"
                    - discount_rate: The rate of the discount applied
                    - discounted_price: The discounted price of the ordered items
                    - discount_type: The type of discount applied: "standard", "rounding" or "percentage"
                    - reference_quote: The reference quote used to determine the discount strategy
                    - order_size: The size of the order: "small", "medium" or "large"
                    - reference_quote: The reference quote used to determine the discount strategy, empty if none.

            """

            # Step 1: Classify order size
            if base_price < 150:
                order_size = "small"
            elif base_price < 500:
                order_size = "medium"
            else:
                order_size = "large"
            
            # Step 2: Search history using item keywords
            search_terms = [k["item_name"] for k in order_items] 
        
             # Add order size to narrow results
            search_terms.append(order_size)
            history = search_quote_history(search_terms, limit=5)
            
            # Step 3: Pick best match - most product keyword overlap 
            
            if not history:
                # Broader search with just order size
                history = search_quote_history([order_size], limit=10)
                
            best_score = 0
            best_history_match = None
            item_keywords = [item.lower() for item in search_terms]
            for quote in history:
                explanation = quote["quote_explanation"].lower()
                score = sum(1 for kw in item_keywords if kw in explanation)
                if score > best_score:
                    best_score = score
                    best_history_match = quote
            
            # If no product overlap, take first (most recent)
            if best_history_match is None and history:
                best_history_match = history[0]
            
            # Step 4: Extract discount strategy
            strategy = {
                "type": "standard",
                "order_size": order_size,
                "discount_rate": {"small": 0.05, "medium": 0.10, "large": 0.15}[order_size],
                "discounted_price" : base_price,
                "round": True,
                "reference_quote": best_history_match["quote_explanation"] if best_history_match else ""
            }
            
            if best_history_match:
                # Check for explicit percentage
                pct_match = re.findall(r'(\d+)%', best_history_match["quote_explanation"])
                if pct_match:
                    pct = int(pct_match[0]) / 100
                    if pct < 0.2: # Ignore percentage discounts higher than 20%
                        strategy["type"] = "percentage"
                        strategy["discount_rate"] = pct
                
                # Check for rounding
                if "round" in best_history_match["quote_explanation"].lower():
                    if strategy["type"] == "standard":
                        strategy["type"] = "rounding"
            
            if strategy["type"] == "rounding":
                strategy["discounted_price"] = 10 * ( base_price // 10 )
                strategy["discount_rate"] = round(1.0 - strategy["discounted_price"]/base_price, 2)

            if strategy["type"] == "percentage" or strategy["discounted_price"] == base_price:
                strategy["type"] = "percentage"
                strategy["discounted_price"] = base_price * (1 - strategy["discount_rate"])
            
            return strategy


        @tool
        def get_quote(inventory_items_order: List[Dict[str, str]]) -> str:
            """
            Get a quote for the ordered items.

            Args:
                inventory_items_order: A list of string dictionaries containing the ordered items with format: [{"item_name":<item_1>, "quantity":<number of items>}, {"item_name":<item_2>, "quantity":<number of items>},...]
            
            Returns:
                A sumary of the order quote and applied discout:
                    - discount_details_per_item: Dictionary containing the discount details of each ordered item:
                        - quantity: The quantity of the ordered item
                        - unit_price_before_discount: The unit price of the ordered item before discount
                        - total_order_price_before_discount: The total price of the ordered item before discount
                        - unit_price_after_discount: The unit price of the ordered item after discount
                        - total_order_price_after_discount: The total price of the ordered item after discount
                    
                    - total_price: The total price of the ordered items before discount
                    - discounted_price: The total price of the ordered items after discount
                    - discount_type: The type of discount applied: "standard", "rounding" or "percentage"
                    - discount_rate: The rate of the discount applied when the discount_type is "percentage" or "rounding".
                    - reference_quote: The reference quote used to determine the discount strategy, empty if none.
            """
            self.quote_result = {}

            total_price = 0.0
            item_catolog_price_dict= {}
            for item in inventory_items_order:
                item_name = item["item_name"]
                quantity = int(item["quantity"])

                item_catolog_price = get_inventory_item_unit_price(item_name)
                raw_order_price =  item_catolog_price * quantity
                item_catolog_price_dict[item_name] = (quantity, item_catolog_price, raw_order_price)

                total_price += raw_order_price
                
            strategy = get_quote_strategy(inventory_items_order, total_price)
            
            # Compute the discounted price for each item and adjust the discounted price accordingly
            # to avoid rounding errors and ensure the total discounted price is correct
            total_discounted_price = 0.
            discount_details_per_item = {}
            highest_item_price = 0.0
            highest_item_name = ""
            for item_name, (quantity, item_catolog_price, raw_order_price) in item_catolog_price_dict.items():
                unit_discounted_price = round(item_catolog_price * (1 - strategy["discount_rate"]), 2)
                item_total_discounted_price = quantity * unit_discounted_price
                if item_total_discounted_price > highest_item_price:
                    highest_item_price = item_total_discounted_price
                    highest_item_name = item_name
                
                total_discounted_price += item_total_discounted_price
                
                discount_details_per_item[item_name] = {
                    "quantity": quantity, 
                    "unit_price_before_discount": item_catolog_price, 
                    "total_order_price_before_discount": raw_order_price, 
                    "unit_price_after_discount": unit_discounted_price, 
                    "total_order_price_after_discount": item_total_discounted_price
                }

            # Adjust the discounted price to ensure the total discounted price is correct
            round_error = strategy["discounted_price"] - total_discounted_price
            if round_error > 0:
                discount_details_per_item[highest_item_name]["total_order_price_after_discount"] += round_error

            self.quote_result = {
                "discount_details_per_item": discount_details_per_item,
                "total_price": total_price,
                "discounted_price": strategy["discounted_price"],
                "discount_type": strategy["type"],
                "discount_rate": strategy["discount_rate"],
                "reference_quote": strategy["reference_quote"]
            }

            return {"status": "success", "message": f"quote_result: {json.dumps(self.quote_result)}"}

        super().__init__(
            tools=[get_quote],
            model=model,
            name="quoting_agent",
            description="""Agent that elaborate discount strategy with regard to 
            the order size and the historical quotes, to generate a quote for the ordered items.
            """,
        )

# Tools for ordering agent
class TransactionsAgent(ToolCallingAgent):
    """Calculate the total price for the ordered items with bulk discounts applied."""
    def __init__(self, model, max_attempts: int = 1):
        self.model = model

        @tool
        def complete_transactions(ordered_items: str, discount_details: dict, discounted_price: int, discout_rate: float, request_date: str, delivery_date: str) -> str:
            """Complete the transactions for the ordered items.
            Args:
                ordered_items: Comma-separated string containing the ordered items with format: <catalog_item_name>:<quantity>:<in_stock_true_or_false>:<stock_reorder_quantity>,...
                
                discount_details: Dictionary with the item_name as key and the discount details as value containing the discount details of each ordered item:
                    - quantity: The quantity of the ordered item
                    - unit_price_before_discount: The unit price of the ordered item before discount
                    - total_order_price_before_discount: The total price of the ordered item before discount
                    - unit_price_after_discount: The unit price of the ordered item after discount
                    - total_order_price_after_discount: The total price of the ordered item after discount
                    Example: {"item_name": "item_1", "quantity": 10, "unit_price_before_discount": 100, "total_order_price_before_discount": 1000, "unit_price_after_discount": 90, "total_order_price_after_discount": 900}
                    
                discounted_price: The discounted price of the ordered items.
                discout_rate: The discount rate applied to the ordered items between 0 and 1.
                request_date: The date of the request in ISO format (YYYY-MM-DD)
                delivery_date: The delivery date in ISO format (YYYY-MM-DD)
            Returns:
                The transaction result with the cash balance and inventory changes if the transactions succeed.
                An error message if the transactions failed.
            """
            print(f">> Discount details: {discount_details}")
            quote_result_list = [item.split(':') for item in ordered_items.split(',')]
            oredered_items_dicts = []
            for item in quote_result_list:
                oredered_items_dicts.append(
                    {
                    "item_name": item[0],
                    "quantity": int(float(item[1])),
                    "in_stock": item[2].lower() == "true",
                    "reorder_quantity": int(float(item[3]))
                    }
                )
            
            # Latest delivery date is the request date + 1 day
            earliest_delivery_date = datetime.fromisoformat(request_date) + timedelta(days=1)
            stock_order_items = {}
            for item in oredered_items_dicts:
                if item["in_stock"]:
                    continue

                stock_order_quantity = item["reorder_quantity"]
                if stock_order_quantity == 0:
                    continue

                stock_order_items[item["item_name"]] = stock_order_quantity
                
                supplier_delivery_date = get_supplier_delivery_date(request_date, stock_order_quantity)
                
                # Convert delivery_date to ISO format and get latest delivery date
                supplier_delivery_date = datetime.fromisoformat(supplier_delivery_date)
                if supplier_delivery_date > earliest_delivery_date :
                    earliest_delivery_date = supplier_delivery_date
            
            customer_delivery_date = datetime.fromisoformat(delivery_date)
            
            # Check if the latest delivery date is compatible with the requested delivery date
            if earliest_delivery_date > customer_delivery_date:
                return {
                    "success": "stop", 
                    "message": f"Error: Order cannot be delivered on time, The earliest delivery date is {earliest_delivery_date} and the requested delivery date is {delivery_date}"
                }

            financial_report_before_transactions = generate_financial_report(request_date)

            total_supplier_cost = 0.0
            stock_order_prices = {}

            for item_name, stock_order_quantity in stock_order_items.items():
                stock_order_price = get_inventory_item_unit_price(item_name) * stock_order_quantity
                stock_order_price = stock_order_price * (1 - SUPPLY_MARGIN_PCT)
                total_supplier_cost += stock_order_price
                stock_order_prices[item_name] = stock_order_price

            # Check if the cash balance is enough to complete the transactions
            current_cash = get_cash_balance(request_date)
            if (current_cash + discounted_price) < total_supplier_cost:
                # This should never happend since the discounted price alone is higher than the total supplier cost.
                # unless the current cash is already negative.
                return {
                    "success": "stop", 
                    "message": f"Error: Cannot complete transactions, due to temporary supply issue."
                }

            # Perform stock_orders transactions
            for item_name, stock_order_price in stock_order_prices.items():
                create_transaction(item_name, "stock_orders", stock_order_items[item_name], stock_order_price, request_date)
                
            # Perform orders transactions
            stock_level_after_transactions = {}
            for item_name, discount_detail in discount_details.items():
                order_quantity = discount_detail["quantity"] 
                item_discounted_price = discount_detail["total_order_price_after_discount"]
                create_transaction(item_name, "sales", order_quantity, item_discounted_price, request_date)
                stock_level_after_transactions[item_name] = get_stock_level(item_name, request_date)
            
            financial_report_after_transactions = generate_financial_report(request_date)
            
            cash_balance_diff = financial_report_after_transactions["cash_balance"] - financial_report_before_transactions["cash_balance"]
            
            # Apply the margin percentage to the unit price.
            inventory_value_diff = SUPPLY_MARGIN_PCT * ( financial_report_after_transactions['inventory_value'] - financial_report_before_transactions['inventory_value'])
        
            return {
                "success": True,
                "message": f"""
                    Transactions have been sucsefully completed for the following items: {oredered_items_dicts} and with earliest delivery date: {earliest_delivery_date}.
                    Cash balance after transactions: {financial_report_after_transactions["cash_balance"]}
                    Cash balannce change after transactions: {cash_balance_diff}
                    Inventory value after transactions: {financial_report_after_transactions["inventory_value"]}
                    Inventory value change after transactions: {inventory_value_diff}
                    stock_level_after_transactions: {stock_level_after_transactions}
                """
            }

        super().__init__(
            tools=[complete_transactions],
            model=model,
            max_steps=max_attempts,
            name="transactions_agent",
            description="""Agent that completes the transactions for the ordered items.""",
        )

# Set up your agents and create an orchestration agent that will manage them.
class OrderRequestOrchestrator(ToolCallingAgent):
    """Orchestrator that coordinates workflow between specialized agents to fullfill a customer order request"""

    def __init__(self, model):
        self.model = model
        self.inventory_agent = InventoryAgent(model)
        self.quoting_agent = QuotingAgent(model)
        # Only one attempt to complete the transactions and not corrupt the database.
        self.transactions_agent = TransactionsAgent(model, max_attempts=1) 
        @tool
        def process_order_request(customer_message: str, order_request_details: List[Dict[str, str]], request_date: str, delivery_date: str) -> str:
            """
            Process the customer order request.
            Args:
                customer_message: A string containing the customer message explaining the order request.
                order_request_details: A list of dictionaries containing the order details of each ordered item format: [{"item_name":<item_1>, "quantity":<number of items>}, {"item_name":<item_2>, "quatity":<number of items>},...]
                request_date: The date of the request in ISO format (YYYY-MM-DD)
                delivery_date: The delivery date in ISO format (YYYY-MM-DD)

            Returns:
                The order process result either succesfull, or failed with an explanation of the failure reason.
            """
            print(f"Order request details received: {order_request_details}")
            # Parse the order request details
            if not delivery_date:
                return {"status": "error", "message": "No delivery date provided"} # NO_DELIVERY_DATE_PROVIDED
            
            # Build the order items dictionary: {item_name: quantity, ...}
            order_items_dict = {}
            for item in order_request_details:
                try:
                    item_name = item["item_name"]
                    quantity = int(item["quantity"])
                    order_items_dict[item_name] = quantity
                except Exception as e:
                    return {
                        "success": False, 
                        "message": f"Error parsing order item: {item['item_name']}"
                    }
                

            catalog_items = get_inventory_item_names()
            
            ## Inventory Agent Call
            inventory_agent_response = self.inventory_agent.run(f"""
                Match customer-requested items to catalog names and check stock availability for each item.

                Customer-requested items: {order_items_dict}
                Catalog items: {catalog_items}
                Request date: {request_date}
                Delivery date: {delivery_date}

                You need to :
                1- Match customer requested items to catalog names using the match_items_to_catalog tool.
                2- Check the stock level of all matched items using the check_stock_level tool.

                CRITICAL: If any tool result status is "stop", you must treat it as an error and immediately call final_answer with using the EXACT message from the tool result. 
                Copy the COMPLETE message word for word. Do NOT summarize, rephrase, or shorten it.
                
                On success, return ONLY the stock status with format: <catalog_item_name>:<ordered_quantity>:<in_stock_true_or_false>:<stock_reorder_quantity>
                """
            )

            print(f">> Inventory Response: {inventory_agent_response}" )

            if inventory_agent_response.strip().lower().startswith("error"):
                return {
                    "success": False, 
                    "message": inventory_agent_response.strip()
                }
            

            inventory_items_order = []
            for item in inventory_agent_response.split(','):
                item_name = item.split(':')[0].strip()
                quantity = int(float(item.split(':')[1].strip()))
                inventory_items_order.append({item_name: quantity})
            

            print(f">> Inventory Items Order: {inventory_items_order}")

            ## Quoting Agent Call
            quoting_agent_response = self.quoting_agent.run(f"""
                Provide quotes for the ordered items.
                Customer message: {customer_message} 
                Ordered items: {inventory_items_order}
                Delivery date: {delivery_date}

                Use the get_quote tool to get the quote for all items, with discount applied to the total amount.
                You MUST call get_quote with these exact parameters:
                - inventory_items_order: [{{"item_name":<item_1>, "quantity":<number of items>}}, {{"item_name":<item_2>, "quantity":<number of items>}},...]
                
                CRITICAL: If any tool result status is "stop", you must immediately call final_answer with that error message starting with "Error:" followed by the reason if the processing did not succeed. Do NOT retry or call any other tool.

                On success, return the quotation message to the customer with:
                - the applied discount strategy details with the discount type and rate,
                - The ordered quantities of each item,
                - the unit price of each item before and after discount,  
                - the total order price before and after discount
                - Use a response tailored to the customer initial request, using an exmaple message style returned by the get_quote tool when provided and specific to the event when mentionned.
                - Do NOT mention any internal details like the discount size (small,...), or the quote reference.
                """
            )
         
            print(f">> Quoting Response: {quoting_agent_response}" )
            if quoting_agent_response.strip().lower().startswith("error"):
                return {
                    "success": False, 
                    "message": quoting_agent_response.strip()
                }

            quote_result = self.quoting_agent.quote_result

            ## Transactions Agent Call
            transaction_message =  f"""
                Process the transaction for the ordered items.
                
                Oredered_items: {inventory_agent_response}

                Discount_details_per_item: {json.dumps(quote_result["discount_details_per_item"])}
                Initial_price: {quote_result["total_price"]}
                Discounted_price: {quote_result["discounted_price"]}
                Discount_type: {quote_result["discount_type"]}
                Discount_rate: {quote_result["discount_rate"]}
 
                Request date: {request_date}
                
                Delivery date: {delivery_date}
                
                Use the complete_transactions tool to create the transaction for the ordered items.

                CRITICAL: If any tool result status is "stop", you must immediately call final_answer with that error message starting with "Error:" followed by the reason if the processing did not succeed. Do NOT retry or call any other tool.

                On success, return the success message with the prefix "Success:" followed by the message.
                """

            transaction_agent_response = self.transactions_agent.run(transaction_message)

            if transaction_agent_response.strip().lower().startswith("error"):
                return {
                    "success": False, 
                    "message": transaction_agent_response.strip()
                }

            return {
                "success": True, 
                "message": quoting_agent_response
            }


        super().__init__(
                tools=[process_order_request],
                model=model,
                name='orchestrator',
                max_steps=3,
                description="""You are an orchestrator that manages customer order request workflow from Beaver's Choice Paper Company.
                You coordinate between inventory, quotation, and sales agents.
                Your focus is on handling efficiently sales transactions.
                """,
            )

def call_your_multi_agent_system(request_with_date)-> str:
    print(f">> Processing order request: {request_with_date}")
    orchestrator = OrderRequestOrchestrator(model)

    response = orchestrator.run(f"""
        Parse the following order request and extract order details.

        CUSTOMER ORDER REQUEST:
        {request_with_date}

        Parse the order request details and extract :
        1. Order items ONLY in format: <item>:<quntity>,<item>:<quntity>
        2. Request date (ISO)
        3. Delivery date (ISO)

        Pass the extrated entities separately to the process_order_request tool.
        You MUST call process_order_request with these exact parameters:
            - customer_message: the original customer text
            - order_request_details: [{{"item_name":<item_1>, "quantity":<number of items>}}, {{"item_name":<item_2>, "quantity":<number of items>}},...]
            - request_date: ISO format (YYYY-MM-DD)
            - delivery_date: ISO format (YYYY-MM-DD)

        
        Return the response as a string with the prefix "Error:" if the processing failed.
        The failure MUST mention non avilable items when applicable.
        The failure MUST never mention the inventory or cash balance details.
        
        CRITICAL: If any tool result status is "error" or "stop", you must treat it as an error and immediately call final_answer with using the error message from the tool result. 

        If success, return a message with the prefix "Success:" followed by an email message to the customer using a narrative style.
        The success message must mention ONLY the delivery date, the ordered items, ordered quantity, the initial unit price, the overall discount, total price before and after discount. 
        Do NOT mention any internal details like the discount size (small,...), or the quote reference.
        
        SUCCES MESSAGE EXAMPLE:
        "Success: Thank you for your order! Here’s the breakdown: 500 sheets of A4 printer paper at $0.05 each totals $25.00; 300 sheets of letter-sized paper at $0.06 each comes to $18.00; and 100 sheets of heavy cardstock at $0.15 each totals $15.00. Since you're ordering in bulk across categories, I'll apply a bulk discount of $5.00, bringing your total to a rounded and tidy $53.00. This ensures you receive the best deal for your event preparation."
        """)
    print(f"Response: {response}")
    print('--------------------------------')
    return response


# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    results = []
    for idx, row in quote_requests_sample.iterrows():

        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############
        
        response = call_your_multi_agent_system(request_with_date)

        if response.strip().lower().startswith("error:"):
            response = response[len("error:"):].strip()
        elif response.strip().lower().startswith("success:"):
            response = response[len("success:"):].strip()


        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()
