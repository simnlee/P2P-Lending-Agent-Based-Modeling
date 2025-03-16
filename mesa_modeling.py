"""
Agent-Based Model for a Peer-to-Peer Lending Market Simulation

This simulation uses the Mesa library to model the behavior of borrowers and lenders
in a decentralized lending market. The model is designed based on the document that
describes the lending platform attributes, agent (borrower and lender) attributes, platform
rules, loan request/approval process, and initial conditions :contentReference[oaicite:1]{index=1}.

Key features include:
    - Dynamic interest rate adjustment based on platform liquidity utilization.
    - Auction-based loan matching between borrowers and lenders.
    - Agent decision-making that incorporates risk assessment, reputation, and liquidity management.
    - Simulation of market volatility that affects repayment behavior.
    - Graphical visualization of key statistics for research analysis.

This environment is designed to help explore how different agent strategies, 
incorporating risk assessment and reputation, impact market stability and efficiency.
"""

import random
import matplotlib.pyplot as plt
from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector

# =============================================================================
# Borrower Agent Class
# =============================================================================
class BorrowerAgent(Agent):
    """
    A borrower agent that decides when to request a loan based on their cash flow,
    existing debt, and risk profile. Borrowers have attributes such as:
       - borrow_balance: Current total debt.
       - collateral_balance: Funds or assets pledged as collateral.
       - reputation_score: Reflects past repayment behavior.
       - income_flow: The ability to generate cash flow for repayments.
       - credit_rating: Categorical risk indicator ('high', 'medium', 'low').
       - health_factor: Computed risk metric that determines liquidation risk.
    
    The decision process is simplified here: if a borrower has low cash reserves
    (simulated via a random chance influenced by credit rating), they submit a loan request.
    """
    def __init__(self, unique_id, model, credit_rating):
        super().__init__(model)  # Explicit call to parent class
        self.credit_rating = credit_rating  # "high", "medium", or "low"
        self.borrow_balance = 0.0  # total amount borrowed
        # For simplicity, collateral is set based on credit rating:
        # High-credit borrowers get better collateral values.
        if self.credit_rating == "high":
            self.collateral_balance = random.uniform(10000, 20000)
            self.reputation_score = random.uniform(0.8, 1.0)
            self.income_flow = random.uniform(500, 1000)
        elif self.credit_rating == "medium":
            self.collateral_balance = random.uniform(5000, 15000)
            self.reputation_score = random.uniform(0.5, 0.8)
            self.income_flow = random.uniform(300, 700)
        else:  # low credit
            self.collateral_balance = random.uniform(1000, 8000)
            self.reputation_score = random.uniform(0.2, 0.5)
            self.income_flow = random.uniform(100, 500)
        self.health_factor = float('inf')  # will be updated when debt > 0
        # To store current loan request (if any)
        self.loan_request = None

    def step(self):
        """
        Each step the borrower decides whether to request a loan.
        The decision is based on comparing their income flow to their debt obligations.
        A random element is added to simulate diverse individual behaviors.
        """
        # Simplified decision: request a loan if no active loan request exists and
        # if a random chance (inversely related to reputation) indicates need.
        if self.loan_request is None and random.random() > self.reputation_score:
            self.request_loan()
        
        # Simulate repayment or default behavior here (simplified).
        # In a more advanced simulation, borrowers would also decide to repay or default.
        self.update_health_factor()

    def request_loan(self):
        """
        Creates a loan request with:
            - amount: chosen as a function of income_flow and collateral.
            - preferred_interest: For stable-rate loans, but here we simply record a flag.
            - collateral: based on collateral_balance.
        
        The request is then added to the model's list of pending loan requests.
        """
        # Determine loan amount based on a fraction of collateral and income.
        loan_amount = min(self.collateral_balance * 0.5, self.income_flow * 10)
        self.loan_request = {
            "amount": loan_amount,
            "preferred_interest": None,  # For simplicity, not setting a fixed rate here.
            "collateral": self.collateral_balance
        }
        # Register the loan request with the model
        self.model.loan_requests.append((self, self.loan_request))
        # Debug: print(f"Borrower {self.unique_id} requested loan of {loan_amount}")

    def update_health_factor(self):
        """
        Health Factor is computed as (collateral * liquidation_threshold) / borrow_balance.
        A value below 1 indicates risk of liquidation.
        """
        if self.borrow_balance > 0:
            self.health_factor = (self.collateral_balance * self.model.liquidation_threshold) / self.borrow_balance
        else:
            self.health_factor = float('inf')
        # Reset loan_request after processing (simulate one-off request)
        if self.loan_request is not None:
            self.loan_request = None

# =============================================================================
# Lender Agent Class
# =============================================================================
class LenderAgent(Agent):
    """
    A lender agent that provides funds for loans. Lenders have attributes such as:
       - lending_balance: Total available funds for lending.
       - risk_tolerance: A value between 0 and 1 indicating willingness to take on risk.
       - interest_rate_preference: Preferred range of interest rates.
       - active_loans: Record of funded loans.
    
    Lenders scan pending loan requests and decide whether to bid for funding.
    Their bidding behavior is influenced by their risk tolerance and interest rate preferences.
    """
    def __init__(self, unique_id, model, risk_tolerance):
        super().__init__(model)  # Correct initialization
        # Starting capital for lending is drawn from a uniform distribution
        self.lending_balance = random.uniform(100000, 500000)
        self.risk_tolerance = risk_tolerance  # Lower values: more risk averse.
        # For simplicity, we set a fixed interest rate range preference:
        self.interest_rate_preference = (self.model.base_rate, self.model.base_rate * 1.5)
        self.active_loans = []  # list of dicts for active loans

    def step(self):
        """
        Each step the lender evaluates available loan requests.
        If a loan request meets the risk criteria (assessed via borrower credit and collateral),
        the lender submits a bid. The bid interest rate is adjusted based on the lender's risk tolerance.
        """
        # Iterate through all pending loan requests
        for request in self.model.loan_requests:
            borrower, loan_request = request
            if self.evaluate_loan(borrower, loan_request):
                # Submit a bid to the platform's auction mechanism.
                self.model.submit_bid(borrower, self, loan_request)
        # Optionally, lenders manage liquidity based on market conditions.
        self.manage_liquidity()

    def evaluate_loan(self, borrower, loan_request):
        """
        Evaluate the loan request based on:
            - Borrower's credit rating and reputation.
            - Collateral provided relative to the requested amount.
            - The platform’s current interest rate versus the lender's preference.
        
        Here, a simplified heuristic is used:
            - High reputation and sufficient collateral favor a bid.
            - A random component simulates differing individual strategies.
        """
        # For simplicity, compare collateral coverage ratio:
        coverage_ratio = loan_request["collateral"] / loan_request["amount"]
        current_rate = self.model.calculate_interest_rate()
        # If coverage ratio is high and the current rate is within lender's acceptable range:
        if coverage_ratio >= 1.5 and self.interest_rate_preference[0] <= current_rate <= self.interest_rate_preference[1]:
            # Additionally, if the borrower's reputation is high or if the lender's risk tolerance is high:
            if borrower.reputation_score > 0.6 or self.risk_tolerance > 0.5:
                return True
        # Otherwise, a random chance (scaled by risk tolerance) can lead to a bid.
        return random.random() < self.risk_tolerance * 0.3

    def manage_liquidity(self):
        """
        Lenders can decide to withdraw or reinvest funds.
        In this simplified model, this function can be expanded to simulate liquidity management.
        """
        # For now, we do not change lending_balance dynamically aside from loan funding.
        pass

# =============================================================================
# Lending Platform Model Class
# =============================================================================
class LendingPlatformModel(Model):
    """
    The main model that simulates the decentralized lending platform.
    
    Key platform attributes include:
       - Total Liquidity: The available funds on the platform.
       - Total Borrowed: Sum of all active loans.
       - Interest Rate Mechanism: A two-slope model that adjusts the interest rate based on the utilization rate.
       - Market Volatility: A parameter that influences borrower repayment behavior and default risk.
    
    The model maintains a list of pending loan requests and collected bids, which are processed
    at each step via an auction mechanism.
    """
    def __init__(self, num_borrowers=1000, num_lenders=200, initial_liquidity=50e6, base_rate=0.02, target_utilization=0.8):
        # Call the parent class (Model) __init__ method to initialize self.random and other attributes.
        super().__init__()
        
        self.num_borrowers = num_borrowers
        self.num_lenders = num_lenders
        self.initial_liquidity = initial_liquidity
        self.base_rate = base_rate
        self.target_utilization = target_utilization
        self.volatility = 0.1  
        self.slope1 = 0.1  # When utilization is below target.
        self.slope2 = 0.3  # When utilization is above target.
        self.liquidation_threshold = 1.0
        
        self.liquidity = initial_liquidity
        self.total_borrowed = 0.0
        
        self.loan_requests = []
        self.bids = []  # Each bid is a tuple: (borrower, lender, loan_request, bid_interest_rate)
        
        # Initialize the random activation schedule.
        self.schedule = RandomActivation(self)
        
        # Create borrower agents with credit distribution as specified.
        for i in range(self.num_borrowers):
            if i < self.num_borrowers * 0.5:
                credit = 'high'
            elif i < self.num_borrowers * 0.8:
                credit = 'medium'
            else:
                credit = 'low'
            borrower = BorrowerAgent(i, self, credit)
            self.schedule.add(borrower)
        
        # Create lender agents with varying risk tolerances.
        for i in range(self.num_lenders):
            risk_tolerance = random.uniform(0, 1)
            lender = LenderAgent(self.num_borrowers + i, self, risk_tolerance)
            self.schedule.add(lender)
        
        # DataCollector for tracking key metrics over time.
        self.datacollector = DataCollector(
            model_reporters={
                "Liquidity": lambda m: m.liquidity,
                "TotalBorrowed": lambda m: m.total_borrowed,
                "Utilization": lambda m: m.total_borrowed / m.liquidity if m.liquidity > 0 else 0,
                "AverageInterestRate": self.calculate_interest_rate
            }
        )

    def calculate_interest_rate(self):
        """
        Compute the current interest rate based on the platform’s utilization of liquidity.
        Uses a two-slope model:
            - If utilization (U) < target, Interest Rate = base_rate + slope1 * U.
            - If U ≥ target, Interest Rate = base_rate + slope1 * target + slope2 * (U - target).
        """
        utilization = self.total_borrowed / self.liquidity if self.liquidity > 0 else 0
        if utilization < self.target_utilization:
            return self.base_rate + self.slope1 * utilization
        else:
            return self.base_rate + self.slope1 * self.target_utilization + self.slope2 * (utilization - self.target_utilization)

    def submit_bid(self, borrower, lender, loan_request):
        """
        Lenders submit bids for loan requests via this function.
        The bid interest rate is calculated as the current platform rate adjusted by the lender's risk profile.
        """
        current_rate = self.calculate_interest_rate()
        # Lenders with lower risk tolerance offer a slightly lower bid rate.
        bid_rate = current_rate * (1 + (1 - lender.risk_tolerance) * 0.05)
        self.bids.append((borrower, lender, loan_request, bid_rate))
        # Debug: print(f"Lender {lender.unique_id} submitted bid at rate {bid_rate:.4f} for Borrower {borrower.unique_id}")

    def match_loans(self):
        """
        Processes all collected bids and matches loan requests using an auction mechanism.
        For each borrower with one or more bids, the lowest bid is selected to fund the loan.
        The platform updates borrower debt, lender funds, and overall borrowed totals.
        """
        # Group bids by borrower.
        bids_by_borrower = {}
        for bid in self.bids:
            borrower, lender, loan_request, bid_rate = bid
            if borrower not in bids_by_borrower:
                bids_by_borrower[borrower] = []
            bids_by_borrower[borrower].append(bid)
        
        # For each borrower, select the bid with the lowest interest rate.
        for borrower, bids in bids_by_borrower.items():
            best_bid = min(bids, key=lambda b: b[3])
            lender = best_bid[1]
            loan_request = best_bid[2]
            bid_rate = best_bid[3]
            loan_amount = loan_request["amount"]
            # Update the borrower’s debt.
            borrower.borrow_balance += loan_amount
            # Update total borrowed funds on the platform.
            self.total_borrowed += loan_amount
            # Lender funds are reduced (assuming funds are locked for the duration of the loan).
            lender.lending_balance -= loan_amount
            # Record the loan in the lender’s active loans.
            lender.active_loans.append({
                'borrower_id': borrower.unique_id,
                'amount': loan_amount,
                'interest_rate': bid_rate
            })
            # (Optional: record the loan in a platform-wide ledger.)
        # Clear all processed loan requests and bids.
        self.loan_requests = []
        self.bids = []

    def step(self):
        """
        At each step:
           - The market volatility index is updated to simulate changing economic conditions.
           - All agents take their step (making loan requests, bidding, etc.).
           - The platform matches loan requests with bids via the auction mechanism.
           - Simulated repayments occur, affecting total borrowed funds and liquidity.
           - Data is collected for later analysis.
        """
        # Update market volatility randomly (this could represent external shocks).
        self.volatility = random.uniform(0.05, 0.2)
        
        # Step all agents.
        self.schedule.step()
        
        # Process all loan bids and match loans.
        self.match_loans()
        
        # Simulate repayments: borrowers repay a fraction of their debt based on volatility.
        repayment_fraction = 0.01 * self.volatility  # higher volatility increases repayment pressure.
        repaid_amount = self.total_borrowed * repayment_fraction
        self.total_borrowed = max(self.total_borrowed - repaid_amount, 0)
        self.liquidity += repaid_amount
        
        # Collect model-level data.
        self.datacollector.collect(self)

# =============================================================================
# Visualization and Simulation Runner
# =============================================================================
def run_simulation(steps):
    """
    Runs the lending platform simulation for a specified number of steps.
    After the simulation, key statistics are plotted:
         - Platform Liquidity over time.
         - Total Borrowed funds over time.
         - Utilization rate over time.
         - Average Interest Rate over time.
    
    Returns the final model state and collected data for further analysis.
    """
    model = LendingPlatformModel()
    for i in range(steps):
        model.step()
    
    # Retrieve data from the DataCollector.
    data = model.datacollector.get_model_vars_dataframe()
    
    # Create visualizations.
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(data["Liquidity"])
    plt.title("Platform Liquidity Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Liquidity")
    
    plt.subplot(2, 2, 2)
    plt.plot(data["TotalBorrowed"])
    plt.title("Total Borrowed Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Total Borrowed")
    
    plt.subplot(2, 2, 3)
    plt.plot(data["Utilization"])
    plt.title("Utilization Rate Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Utilization Rate")
    
    plt.subplot(2, 2, 4)
    plt.plot(data["AverageInterestRate"])
    plt.title("Average Interest Rate Over Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Interest Rate")
    
    plt.tight_layout()
    plt.show()
    
    return model, data

# =============================================================================
# Main Execution
# =============================================================================
if __name__ == '__main__':
    # Run the simulation for X steps.
    model, data = run_simulation(2000)