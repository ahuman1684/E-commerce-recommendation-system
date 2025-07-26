from flask import Blueprint, render_template, request, session
from app.model import RecommendationModel
import matplotlib
import matplotlib.pyplot as plt
import os

# Define Blueprint
app_routes = Blueprint('app_routes', __name__)

# Load recommendation model
model = RecommendationModel(data_path="data/fashion.csv")
model.build_recommendations()
matplotlib.use('Agg')

# Home page route
@app_routes.route("/")
def home():
    # Pagination setup
    page = int(request.args.get("page", 1))
    items_per_page = 10

    start = (page - 1) * items_per_page
    end = start + items_per_page

    total_items = len(model.products)
    products = model.products.iloc[start:end].to_dict(orient='records')

    has_next = end < total_items
    has_previous = start > 0

    # Get top-selling products
    top_selling = model.products.sort_values("sales", ascending=False).head(5).to_dict(orient='records')

    return render_template(
        "home.html",
        products=products,
        top_selling=top_selling,
        page=page,
        has_next=has_next,
        has_previous=has_previous
    )

# Save browsed products in the session
@app_routes.route("/product/<int:product_id>")
def product_details(product_id):
    # Fetch product details
    product = model.products.loc[model.products['product_id'] == product_id].to_dict('records')[0]

    # Get recommendations
    recommendations = model.recommend(product_id)

    # Store viewed products in session
    viewed_products = session.get("viewed_products", [])
    if product_id not in viewed_products:
        viewed_products.append(product_id)
        session["viewed_products"] = viewed_products

    return render_template(
        "product_details.html",
        product=product,
        recommendations=recommendations
    )

# Route for recommendations based on user history
@app_routes.route("/recommendations")
def for_you():
    viewed_products = session.get("viewed_products", [])
    recommendations = []
    
    if viewed_products:
        last_viewed_product = viewed_products[-1]
        recommendations = model.recommend(last_viewed_product)

    return render_template("for_you.html", recommendations=recommendations)


# Route for model performance dashboard
@app_routes.route("/dashboard")
def model_dashboard():
    # Example ground truth (can be replaced with actual user data)
    ground_truth = {
        58352: [58354, 15970, 39386],
        15970: [58352, 58354, 39386],
    }

    precision_list = []
    recall_list = []

    for product_id, relevant_items in ground_truth.items():
        # Model recommendations for this product
        recommendations = [rec["product_id"] for rec in model.recommend(product_id)]

        # Calculate Precision and Recall
        relevant_recommended = set(recommendations) & set(relevant_items)
        precision = len(relevant_recommended) / len(recommendations) if recommendations else 0
        recall = len(relevant_recommended) / len(relevant_items) if relevant_items else 0

        precision_list.append(precision)
        recall_list.append(recall)

    # Average Precision and Recall
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0
    avg_recall = sum(recall_list) / len(recall_list) if recall_list else 0

    # F1 Score
    f1_score = (
        2 * avg_precision * avg_recall / (avg_precision + avg_recall)
        if avg_precision + avg_recall > 0
        else 0
    )

    # Total products and recommendations
    total_products = len(model.products)
    total_recommendations = len(ground_truth)

    # Category distribution
    category_distribution = model.products["category"].value_counts().to_dict()

    return render_template(
        "model_dashboard.html",
        total_products=total_products,
        total_recommendations=total_recommendations,
        avg_precision=avg_precision,
        avg_recall=avg_recall,
        f1_score=f1_score,
        category_distribution=category_distribution,
    )
