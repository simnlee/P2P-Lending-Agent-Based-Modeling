from mesa import Model
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from agents import LenderAgent, BorrowerAgent
import numpy as np

class AAVELendingModel(Model):
    """A model of the AAVE lending protocol with reputation-based collateral factors."""
    
    def __init__(
        self,
        num_lenders=50,
        num_borrowers=100,
        base_collateral_factor=0.75,
        reputation_sensitivity=0.2,
        liquidation_threshold=1.05,
        base_interest_rate=0.03,
        seed=None
    ):
        super().__init__()
        self.running = True
        self.num_lenders = num_lenders
        self.num_borrowers = num_borrowers
        self.base_collateral_factor = base_collateral_factor
        self.reputation_sensitivity = reputation_sensitivity
        self.base_liquidation_threshold = liquidation_threshold
        self.base_interest_rate = base_interest_rate
        
        # Initialize the scheduler
        self.schedule = RandomActivation(self)
        
        # Protocol state variables
        self.total_liquidity = 0
        self.total_borrowed = 0
        self.total_liquidations = 0
        self.cumulative_interest = 0
        
        # Market volatility (to simulate price changes)
        self.market_volatility = 0.1
        
        print(f"Initializing model with reputation sensitivity: {reputation_sensitivity}")
        
        # Create lenders
        for i in range(self.num_lenders):
            lender = LenderAgent(i, self)
            self.schedule.add(lender)
            
        # Create borrowers
        for i in range(self.num_borrowers):
            borrower = BorrowerAgent(self.num_lenders + i, self)
            self.schedule.add(borrower)
        
        print(f"Initial liquidity: {self.total_liquidity:.2f}")
        
        # Initialize data collector
        self.datacollector = DataCollector(
            model_reporters={
                "Utilization_Rate": lambda m: m.get_utilization_rate(),
                "Total_Liquidity": lambda m: m.total_liquidity,
                "Total_Borrowed": lambda m: m.total_borrowed,
                "Total_Liquidations": lambda m: m.total_liquidations,
                "Average_Reputation": self.get_average_reputation,
                "Average_Collateral_Ratio": self.get_average_collateral_ratio
            },
            agent_reporters={
                "Reputation": lambda a: getattr(a, "reputation_score", None),
                "Borrowed_Amount": lambda a: getattr(a, "borrowed_amount", 0),
                "Collateral": lambda a: getattr(a, "collateral", 0),
                "Is_Liquidated": lambda a: getattr(a, "is_liquidated", False)
            }
        )
    
    def get_utilization_rate(self):
        """Calculate the current utilization rate of the protocol."""
        if self.total_liquidity == 0:
            return 0
        return self.total_borrowed / self.total_liquidity
    
    def get_average_reputation(self):
        """Calculate the average reputation score of non-liquidated borrowers."""
        borrowers = [agent for agent in self.schedule.agents 
                    if isinstance(agent, BorrowerAgent) and not agent.is_liquidated]
        if not borrowers:
            return 0
        return np.mean([b.reputation_score for b in borrowers])
    
    def get_average_collateral_ratio(self):
        """Calculate the average collateral ratio of active loans."""
        borrowers = [agent for agent in self.schedule.agents 
                    if isinstance(agent, BorrowerAgent) and agent.borrowed_amount > 0]
        if not borrowers:
            return 0
        ratios = [b.collateral / b.borrowed_amount for b in borrowers]
        return np.mean(ratios)
    
    def get_available_liquidity(self):
        """Get the amount of liquidity available for borrowing."""
        return max(0, self.total_liquidity - self.total_borrowed)
    
    def add_liquidity(self, amount):
        """Add liquidity to the protocol."""
        self.total_liquidity += amount
    
    def remove_liquidity(self, amount):
        """Remove liquidity from the protocol."""
        available = self.total_liquidity - self.total_borrowed
        amount = min(amount, available)  # Can't remove more than available
        self.total_liquidity -= amount
        return amount
    
    def process_borrow(self, amount):
        """Process a borrow request."""
        available = self.get_available_liquidity()
        if amount <= available:
            self.total_borrowed += amount
            return True
        return False
    
    def process_repayment(self, amount):
        """Process a loan repayment."""
        self.total_borrowed = max(0, self.total_borrowed - amount)
    
    def liquidate_position(self, borrower):
        """Liquidate a borrower's position."""
        self.total_liquidations += 1
        self.total_borrowed -= borrower.borrowed_amount
        
        # Add liquidation penalty
        penalty = borrower.collateral * 0.1  # 10% liquidation penalty
        self.total_liquidity += penalty
        
        print(f"Liquidation: borrowed={borrower.borrowed_amount:.2f}, collateral={borrower.collateral:.2f}")
        
        # Reset borrower state
        borrower.borrowed_amount = 0
    
    def calculate_lender_interest(self, supplied_amount):
        """Calculate interest earned by a lender."""
        utilization_rate = self.get_utilization_rate()
        
        # Different interest rate models for control vs. reputation-based systems
        if self.reputation_sensitivity == 0:
            # Control case: Lower interest rates at high utilization to discourage lending
            interest_rate = self.base_interest_rate * (1 + utilization_rate)
        else:
            # Reputation-based: Higher interest rates at high utilization to encourage lending
            interest_rate = self.base_interest_rate * (1 + utilization_rate * 3)
            
        interest = supplied_amount * interest_rate / 365  # Daily interest
        self.cumulative_interest += interest
        return interest
    
    def get_liquidation_threshold(self, borrower):
        """
        Calculate the liquidation threshold for a specific borrower based on their reputation.
        Higher reputation borrowers get a slightly lower liquidation threshold (more leeway).
        """
        if self.reputation_sensitivity == 0:
            # In control case, use a higher fixed liquidation threshold (more conservative)
            return 1.8  # Higher threshold means borrowers need to maintain higher collateral ratios
        
        # Adjust threshold based on reputation - higher reputation = lower threshold
        # This means high-reputation borrowers can maintain lower collateral ratios
        reputation_adjustment = self.reputation_sensitivity * 0.2 * (borrower.reputation_score - 0.5)
        adjusted_threshold = max(1.01, self.base_liquidation_threshold - reputation_adjustment)
        
        return adjusted_threshold
    
    def step(self):
        """Advance the model by one step."""
        self.schedule.step()
        self.datacollector.collect(self)
        
        # Market stabilization mechanism
        # If utilization is too low, incentivize borrowing by lowering interest rates
        # If utilization is too high, incentivize lending by increasing interest rates
        utilization_rate = self.get_utilization_rate()
        
        # For reputation-based systems, target a higher utilization rate
        if self.reputation_sensitivity == 0:
            target_utilization = 0.5  # Lower target for control case
        else:
            target_utilization = 0.6 + (self.reputation_sensitivity * 0.1)  # Higher target for reputation systems
        
        # Adjust base interest rate to stabilize utilization
        if utilization_rate < target_utilization - 0.1:
            self.base_interest_rate = max(0.01, self.base_interest_rate * 0.95)  # Decrease interest rate
        elif utilization_rate > target_utilization + 0.1:
            self.base_interest_rate = min(0.1, self.base_interest_rate * 1.05)  # Increase interest rate
        
        # Debug information every 30 steps
        if self.schedule.steps % 30 == 0:
            print(f"Step {self.schedule.steps}:")
            print(f"  Total Liquidity: {self.total_liquidity:.2f}")
            print(f"  Total Borrowed: {self.total_borrowed:.2f}")
            print(f"  Utilization Rate: {self.get_utilization_rate():.2f}")
            print(f"  Base Interest Rate: {self.base_interest_rate:.4f}")
            print(f"  Total Liquidations: {self.total_liquidations}")
            
            # Count active borrowers
            active_borrowers = sum(1 for agent in self.schedule.agents 
                                  if isinstance(agent, BorrowerAgent) 
                                  and agent.borrowed_amount > 0)
            print(f"  Active Borrowers: {active_borrowers}")
            
            # Count liquidated borrowers
            liquidated = sum(1 for agent in self.schedule.agents 
                            if isinstance(agent, BorrowerAgent) 
                            and agent.is_liquidated)
            print(f"  Liquidated Borrowers: {liquidated}")
            
            # Average reputation
            print(f"  Average Reputation: {self.get_average_reputation():.2f}")
            print("") 