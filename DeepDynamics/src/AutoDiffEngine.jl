module AutoDiffEngine

export Node, add, multiply, forward, backward

mutable struct Node
    value::Float64
    grad::Float64
    parents::Vector{Tuple{Node, Float64}}
end

function forward(node::Node)::Float64
    return node.value
end

function backward(node::Node, grad::Float64=1.0)
    node.grad += grad
    for (parent, weight) in node.parents
        backward(parent, grad * weight)
    end
end

function add(n1::Node, n2::Node)::Node
    Node(n1.value + n2.value, 0.0, [(n1, 1.0), (n2, 1.0)])
end

function multiply(n1::Node, n2::Node)::Node
    Node(n1.value * n2.value, 0.0, [(n1, n2.value), (n2, n1.value)])
end

end
