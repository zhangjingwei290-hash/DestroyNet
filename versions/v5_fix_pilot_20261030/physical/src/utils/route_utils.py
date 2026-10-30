"""Route packing and unpacking utilities."""

from typing import List
from ..core.protocol import Route, Solution


def pack_routes(solution: Solution) -> List[List[int]]:
    """
    Pack solution into a list of customer lists.

    Args:
        solution: Solution to pack

    Returns:
        List of routes, where each route is a list of customer IDs
    """
    return [route.customers.copy() for route in solution.routes]


def unpack_routes(packed_routes: List[List[int]]) -> Solution:
    """
    Unpack a list of customer lists into a Solution.

    Args:
        packed_routes: List of routes, where each route is a list of customer IDs

    Returns:
        Solution object
    """
    routes = [Route(customers=list(customer_list)) for customer_list in packed_routes]
    return Solution(routes=routes)


def validate_route_repack(solution: Solution) -> bool:
    """
    Verify that packing and unpacking preserves complete route semantics.

    Args:
        solution: Solution to validate

    Returns:
        True if pack/unpack is reversible
    """
    # Pack
    packed = pack_routes(solution)

    # Unpack
    unpacked = unpack_routes(packed)

    # Compare
    if len(solution.routes) != len(unpacked.routes):
        return False

    for orig_route, new_route in zip(solution.routes, unpacked.routes):
        if orig_route.customers != new_route.customers:
            return False

    # Check customer sets match
    orig_customers = set(solution.get_all_customers())
    new_customers = set(unpacked.get_all_customers())

    return orig_customers == new_customers


def routes_to_compact_string(solution: Solution) -> str:
    """
    Convert solution to compact string representation.

    Format: "route1|route2|route3" where each route is "c1,c2,c3"
    Example: "1,3,5|2,4|6,7,8"

    Args:
        solution: Solution to convert

    Returns:
        Compact string representation
    """
    route_strs = []
    for route in solution.routes:
        if len(route.customers) > 0:
            route_str = ",".join(str(c) for c in route.customers)
            route_strs.append(route_str)

    return "|".join(route_strs)


def compact_string_to_routes(s: str) -> Solution:
    """
    Parse compact string representation into Solution.

    Args:
        s: Compact string (e.g., "1,3,5|2,4|6,7,8")

    Returns:
        Solution object
    """
    if not s or s.strip() == "":
        return Solution(routes=[])

    route_strs = s.split("|")
    routes = []

    for route_str in route_strs:
        route_str = route_str.strip()
        if route_str:
            customers = [int(c) for c in route_str.split(",")]
            routes.append(Route(customers=customers))

    return Solution(routes=routes)
