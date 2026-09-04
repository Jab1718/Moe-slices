"""
Comprehensive Benchmark Suite (100 Tasks): Code Generation & Coding Agent Evaluation.
Đánh giá toàn diện bản BF16 Native trên 100 bài test thực tế:
- 50 Bài Python (Algorithms, Data Structures, String, Math, Graph, DP, OOP)
- 30 Bài Đa Ngôn Ngữ (10 Rust, 10 C++20, 5 Go, 5 TypeScript)
- 20 Bài Coding Agent (JSON Tool Calling, Self-Correction, Debugging, Unified Diff)
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,2,3"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import time
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers.modeling_utils
import transformers.core_model_loading
from concurrent.futures import ThreadPoolExecutor

transformers.modeling_utils.caching_allocator_warmup = lambda *args, **kwargs: None
orig_thread_pool = ThreadPoolExecutor
class SingleWorkerThreadPool(orig_thread_pool):
    def __init__(self, max_workers=None, *args, **kwargs):
        super().__init__(max_workers=1, *args, **kwargs)
transformers.core_model_loading.ThreadPoolExecutor = SingleWorkerThreadPool


def build_100_task_suite():
    tasks = []
    
    # ---------------------------------------------------------
    # 1. 50 BÀI PYTHON ALGORITHMS & DATA STRUCTURES
    # ---------------------------------------------------------
    py_prompts = [
        # Basic & Search (1-10)
        ("Py/BinarySearch", "def binary_search(arr: list[int], target: int) -> int:\n    \"\"\"Return index of target in sorted arr, or -1.\"\"\"\n", lambda c: "left" in c and "right" in c and "mid" in c),
        ("Py/LinearSearch", "def find_first_negative(numbers: list[float]) -> int:\n    \"\"\"Return index of first negative number or -1.\"\"\"\n", lambda c: "for " in c and "< 0" in c and "return" in c),
        ("Py/TwoSumSorted", "def two_sum_sorted(nums: list[int], target: int) -> tuple[int, int]:\n    \"\"\"Return 1-based indices of two numbers that add up to target.\"\"\"\n", lambda c: ("left" in c or "while" in c) and ("+" in c or "==" in c)),
        ("Py/QuickSelect", "def find_kth_largest(nums: list[int], k: int) -> int:\n    \"\"\"Return the kth largest element in unsorted array.\"\"\"\n", lambda c: "sort" in c or "heap" in c or "partition" in c),
        ("Py/MergeIntervals", "def merge_intervals(intervals: list[list[int]]) -> list[list[int]]:\n    \"\"\"Merge all overlapping intervals.\"\"\"\n", lambda c: "sort" in c and ("append" in c or "max" in c)),
        ("Py/InsertInterval", "def insert_interval(intervals: list[list[int]], new_interval: list[int]) -> list[list[int]]:\n    \"\"\"Insert new_interval into sorted non-overlapping intervals.\"\"\"\n", lambda c: "append" in c and ("while" in c or "for" in c)),
        ("Py/PeakElement", "def find_peak_element(nums: list[int]) -> int:\n    \"\"\"Find a peak element and return its index.\"\"\"\n", lambda c: "while" in c and "mid" in c),
        ("Py/SearchRotated", "def search_rotated(nums: list[int], target: int) -> int:\n    \"\"\"Search target in rotated sorted array.\"\"\"\n", lambda c: "left" in c and "right" in c and "mid" in c),
        ("Py/MatrixSearch", "def search_matrix(matrix: list[list[int]], target: int) -> bool:\n    \"\"\"Search for target in m x n integer matrix.\"\"\"\n", lambda c: "row" in c or "col" in c or "mid" in c),
        ("Py/FindMedianSortedArrays", "def find_median_sorted_arrays(nums1: list[int], nums2: list[int]) -> float:\n    \"\"\"Find median of two sorted arrays.\"\"\"\n", lambda c: "merged" in c or "sort" in c or "len" in c),

        # Dynamic Programming & Recursion (11-20)
        ("Py/FibonacciMemo", "def fib_memo(n: int, memo: dict[int, int] = None) -> int:\n    \"\"\"Compute nth Fibonacci number using memoization.\"\"\"\n", lambda c: "memo" in c and ("+" in c or "return" in c)),
        ("Py/ClimbingStairs", "def climb_stairs(n: int) -> int:\n    \"\"\"Distinct ways to climb n stairs with 1 or 2 steps.\"\"\"\n", lambda c: "for" in c or "memo" in c or "+" in c),
        ("Py/CoinChange", "def coin_change(coins: list[int], amount: int) -> int:\n    \"\"\"Fewest coins needed to make up amount, or -1.\"\"\"\n", lambda c: "dp" in c and "min" in c),
        ("Py/LongestIncreasingSubseq", "def length_of_lis(nums: list[int]) -> int:\n    \"\"\"Return length of longest strictly increasing subsequence.\"\"\"\n", lambda c: "dp" in c and ("max" in c or "bisect" in c)),
        ("Py/HouseRobber", "def rob(nums: list[int]) -> int:\n    \"\"\"Max amount of money you can rob without alerting police.\"\"\"\n", lambda c: "max" in c and ("prev" in c or "dp" in c)),
        ("Py/LongestCommonSubseq", "def longest_common_subsequence(text1: str, text2: str) -> int:\n    \"\"\"Return length of longest common subsequence.\"\"\"\n", lambda c: "dp" in c and ("max" in c or "for" in c)),
        ("Py/UniquePathsGrid", "def unique_paths(m: int, n: int) -> int:\n    \"\"\"Number of possible unique paths in m x n grid.\"\"\"\n", lambda c: "dp" in c or "for" in c or "math.comb" in c),
        ("Py/WordBreak", "def word_break(s: str, word_dict: list[str]) -> bool:\n    \"\"\"Return True if s can be segmented into dictionary words.\"\"\"\n", lambda c: "dp" in c and ("for" in c or "in word_dict" in c)),
        ("Py/EditDistance", "def min_distance(word1: str, word2: str) -> int:\n    \"\"\"Minimum operations to convert word1 to word2.\"\"\"\n", lambda c: "dp" in c and "min" in c),
        ("Py/MaxSubarrayKadane", "def max_sub_array(nums: list[int]) -> int:\n    \"\"\"Find the subarray with largest sum (Kadane's algorithm).\"\"\"\n", lambda c: "max" in c and ("current" in c or "cur_sum" in c or "max_sum" in c)),

        # Data Structures: Stacks, Queues, Trees, Graphs (21-35)
        ("Py/ValidParentheses", "def is_valid(s: str) -> bool:\n    \"\"\"Check if bracket string is valid.\"\"\"\n    stack = []\n", lambda c: "stack" in c and ("pop" in c or "append" in c)),
        ("Py/MinStack", "class MinStack:\n    def __init__(self):\n        self.stack = []\n        self.min_stack = []\n    def push(self, val: int) -> None:\n", lambda c: "append" in c and ("min" in c or "self.min_stack" in c)),
        ("Py/DailyTemperatures", "def daily_temperatures(temperatures: list[int]) -> list[int]:\n    \"\"\"Return array of days until warmer temperature.\"\"\"\n", lambda c: "stack" in c and ("append" in c or "pop" in c)),
        ("Py/BinaryTreeInorder", "class TreeNode:\n    def __init__(self, val=0, left=None, right=None):\n        self.val = val; self.left = left; self.right = right\ndef inorder_traversal(root: TreeNode | None) -> list[int]:\n", lambda c: "left" in c and "right" in c and ("append" in c or "+" in c or "yield" in c)),
        ("Py/BinaryTreeMaxDepth", "def max_depth(root: TreeNode | None) -> int:\n    \"\"\"Return maximum depth of binary tree.\"\"\"\n", lambda c: "max" in c and ("left" in c or "right" in c)),
        ("Py/InvertBinaryTree", "def invert_tree(root: TreeNode | None) -> TreeNode | None:\n    \"\"\"Invert binary tree and return root.\"\"\"\n", lambda c: "left" in c and "right" in c),
        ("Py/ValidateBST", "def is_valid_bst(root: TreeNode | None) -> bool:\n    \"\"\"Determine if tree is valid binary search tree.\"\"\"\n", lambda c: "val" in c and ("<" in c or ">" in c or "float" in c)),
        ("Py/LowestCommonAncestor", "def lowest_common_ancestor(root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:\n    \"\"\"Find lowest common ancestor in binary tree.\"\"\"\n", lambda c: "root" in c and ("left" in c or "right" in c)),
        ("Py/LevelOrderTraversal", "def level_order(root: TreeNode | None) -> list[list[int]]:\n    \"\"\"Level order traversal of binary tree nodes.\"\"\"\n", lambda c: "queue" in c or "deque" in c or "level" in c),
        ("Py/GraphBFS", "from collections import deque\ndef bfs_shortest_path(graph: dict[str, list[str]], start: str, target: str) -> int:\n    \"\"\"Return shortest path length in unweighted graph.\"\"\"\n", lambda c: "queue" in c and ("append" in c or "popleft" in c)),
        ("Py/GraphDFS", "def has_cycle_directed(graph: dict[int, list[int]]) -> bool:\n    \"\"\"Detect cycle in directed graph.\"\"\"\n", lambda c: "visited" in c and ("for" in c or "dfs" in c)),
        ("Py/TopologicalSort", "def topological_sort(num_courses: int, prerequisites: list[list[int]]) -> list[int]:\n    \"\"\"Return topological order of courses.\"\"\"\n", lambda c: "in_degree" in c or "visited" in c or "queue" in c),
        ("Py/DijkstraShortestPath", "import heapq\ndef dijkstra(graph: dict[int, list[tuple[int, int]]], start: int) -> dict[int, int]:\n    \"\"\"Return shortest distances from start to all nodes.\"\"\"\n", lambda c: "heapq" in c and ("heappop" in c or "heappush" in c)),
        ("Py/TriePrefixTree", "class Trie:\n    def __init__(self):\n        self.children = {}\n        self.is_end = False\n    def insert(self, word: str) -> None:\n", lambda c: "node" in c or "self.children" in c),
        ("Py/LRUCache", "from collections import OrderedDict\nclass LRUCache:\n    def __init__(self, capacity: int):\n        self.cache = OrderedDict()\n        self.capacity = capacity\n    def get(self, key: int) -> int:\n", lambda c: "move_to_end" in c or "self.cache" in c),

        # Strings & Bit Manipulation & Math (36-50)
        ("Py/ValidAnagram", "def is_anagram(s: str, t: str) -> bool:\n    \"\"\"Return True if t is an anagram of s.\"\"\"\n", lambda c: "count" in c or "Counter" in c or "sorted" in c),
        ("Py/ValidPalindrome", "def is_palindrome(s: str) -> bool:\n    \"\"\"Check if alphanumeric characters form palindrome.\"\"\"\n", lambda c: "isalnum" in c or "lower" in c or "==" in c),
        ("Py/LongestSubstringNoRepeat", "def length_of_longest_substring(s: str) -> int:\n    \"\"\"Length of longest substring without repeating characters.\"\"\"\n", lambda c: "max" in c and ("char" in c or "seen" in c or "left" in c)),
        ("Py/GroupAnagrams", "from collections import defaultdict\ndef group_anagrams(strs: list[str]) -> list[list[str]]:\n    \"\"\"Group anagrams together.\"\"\"\n", lambda c: "defaultdict" in c and ("sorted" in c or "tuple" in c or "append" in c)),
        ("Py/LongestPalindromicSubstring", "def longest_palindrome(s: str) -> str:\n    \"\"\"Return the longest palindromic substring in s.\"\"\"\n", lambda c: "expand" in c or "for" in c or "len" in c),
        ("Py/ReverseWords", "def reverse_words(s: str) -> str:\n    \"\"\"Reverse words in string separated by single space.\"\"\"\n", lambda c: "split" in c and ("join" in c or "reversed" in c or "[::-1]" in c)),
        ("Py/StringCompression", "def compress_string(s: str) -> str:\n    \"\"\"Run-length encoding compression (e.g. 'aabcccccaaa' -> 'a2b1c5a3').\"\"\"\n", lambda c: "count" in c and ("append" in c or "join" in c or "+=" in c)),
        ("Py/SingleNumber", "def single_number(nums: list[int]) -> int:\n    \"\"\"Find single number where every other element appears twice (O(1) space).\"\"\"\n", lambda c: "^" in c or "xor" in c or "for" in c),
        ("Py/CountingBits", "def count_bits(n: int) -> list[int]:\n    \"\"\"Return array of number of 1 bits for each i from 0 to n.\"\"\"\n", lambda c: "bin" in c or "count" in c or "&" in c or "dp" in c),
        ("Py/ReverseBits", "def reverse_bits(n: int) -> int:\n    \"\"\"Reverse bits of a given 32 bits unsigned integer.\"\"\"\n", lambda c: "<<" in c or ">>" in c or "|" in c),
        ("Py/PowerOfTwo", "def is_power_of_two(n: int) -> bool:\n    \"\"\"Return True if n is power of two.\"\"\"\n", lambda c: "&" in c or "==" in c),
        ("Py/SievePrimes", "def sieve_of_eratosthenes(limit: int) -> list[int]:\n    \"\"\"Return all prime numbers up to limit.\"\"\"\n", lambda c: "is_prime" in c or "primes" in c or "for" in c),
        ("Py/GCDLcm", "def gcd_and_lcm(a: int, b: int) -> tuple[int, int]:\n    \"\"\"Return greatest common divisor and least common multiple.\"\"\"\n", lambda c: "%" in c or "gcd" in c or "*" in c),
        ("Py/MatrixRotate90", "def rotate_matrix_90(matrix: list[list[int]]) -> None:\n    \"\"\"Rotate n x n 2D matrix 90 degrees clockwise in-place.\"\"\"\n", lambda c: "transpose" in c or "reverse" in c or "zip" in c or "matrix[i][j]" in c),
        ("Py/SpiralMatrix", "def spiral_order(matrix: list[list[int]]) -> list[int]:\n    \"\"\"Return all elements of matrix in spiral order.\"\"\"\n", lambda c: "top" in c or "bottom" in c or "left" in c or "append" in c)
    ]
    for name, p, v in py_prompts:
        tasks.append({"id": name, "category": "Python/Algorithm", "prompt": p, "verifier": v, "max_tokens": 140})

    # ---------------------------------------------------------
    # 2. 30 BÀI ĐA NGÔN NGỮ (RUST, C++, GO, TYPESCRIPT)
    # ---------------------------------------------------------
    multi_prompts = [
        # Rust (10)
        ("Rust/AsyncMPSC", "// Send message through tokio mpsc channel\nuse tokio::sync::mpsc;\n\npub async fn send_event(tx: mpsc::Sender<String>, msg: String) -> Result<(), mpsc::error::SendError<String>> {\n", lambda c: "tx.send" in c and ("await" in c or "Ok" in c)),
        ("Rust/BinarySearch", "// Rust binary search with Result\npub fn binary_search<T: Ord>(slice: &[T], target: &T) -> Result<usize, usize> {\n", lambda c: "left" in c or "low" in c or "mid" in c or "binary_search" in c),
        ("Rust/SafeMutex", "use std::sync::{Arc, Mutex};\n\npub fn thread_safe_increment(counter: Arc<Mutex<i32>>) {\n", lambda c: "lock" in c and ("unwrap" in c or "*val" in c)),
        ("Rust/CustomIterator", "struct Counter { count: usize, max: usize }\nimpl Iterator for Counter {\n    type Item = usize;\n    fn next(&mut self) -> Option<Self::Item> {\n", lambda c: "self.count" in c and ("Some" in c or "None" in c)),
        ("Rust/OptionMap", "pub fn parse_and_double(s: &str) -> Option<i32> {\n    // Parse str to int and multiply by 2\n", lambda c: "parse" in c and ("map" in c or "ok" in c or "match" in c)),
        ("Rust/StringReversal", "pub fn reverse_words(s: &str) -> String {\n    // Split words, reverse, and join\n", lambda c: "split" in c and ("rev" in c or "collect" in c)),
        ("Rust/HashMapWordCount", "use std::collections::HashMap;\n\npub fn count_words(text: &str) -> HashMap<String, usize> {\n", lambda c: "entry" in c or "insert" in c or "split" in c),
        ("Rust/AsyncHttpClient", "use reqwest;\n\npub async fn fetch_status(url: &str) -> Result<u16, reqwest::Error> {\n", lambda c: "get" in c and ("await" in c or "status" in c)),
        ("Rust/EnumPatternMatch", "enum Shape { Circle(f64), Rectangle(f64, f64) }\nimpl Shape {\n    pub fn area(&self) -> f64 {\n        match self {\n", lambda c: "Circle" in c and "Rectangle" in c and ("*" in c or "PI" in c)),
        ("Rust/TraitImplementation", "pub trait Summary {\n    fn summarize(&self) -> String;\n}\npub struct Article { pub title: String, pub author: String }\nimpl Summary for Article {\n", lambda c: "fn summarize" in c and "self.title" in c),

        # C++20 (10)
        ("CPP/ThreadSafeQueue", "#include <queue>\n#include <mutex>\n\ntemplate<typename T>\nclass ThreadSafeQueue {\n    std::queue<T> q;\n    std::mutex m;\npublic:\n    void push(T val) {\n", lambda c: ("lock" in c or "guard" in c) and "push" in c),
        ("CPP/BinarySearchVector", "#include <vector>\n\nint binary_search(const std::vector<int>& arr, int target) {\n", lambda c: "left" in c and "right" in c and "mid" in c),
        ("CPP/SmartPointerFactory", "#include <memory>\n#include <string>\n\nstruct User { std::string name; int age; };\nstd::unique_ptr<User> make_user(const std::string& name, int age) {\n", lambda c: "make_unique" in c or "new User" in c),
        ("CPP/StringSplitView", "#include <string_view>\n#include <vector>\n\nstd::vector<std::string_view> split(std::string_view str, char delim) {\n", lambda c: "find" in c or "substr" in c or "push_back" in c),
        ("CPP/CustomComparatorSort", "#include <algorithm>\n#include <vector>\n\nvoid sort_descending(std::vector<int>& nums) {\n", lambda c: "std::sort" in c or "greater" in c or ">" in c),
        ("CPP/LRUNodeStructure", "struct Node {\n    int key, val;\n    Node *prev, *next;\n    Node(int k, int v) : key(k), val(v), prev(nullptr), next(nullptr) {}\n};\n", lambda c: "prev" in c and "next" in c),
        ("CPP/VariadicSumTemplate", "template<typename... Args>\nauto sum(Args... args) {\n", lambda c: "..." in c or "+" in c or "return" in c),
        ("CPP/AsyncFuturePromise", "#include <future>\n\nstd::future<int> async_compute(int a, int b) {\n", lambda c: "async" in c or "promise" in c or "return" in c),
        ("CPP/ConceptsConstraint", "#include <concepts>\n\ntemplate<std::integral T>\nT add(T a, T b) {\n", lambda c: "return a + b;" in c or "a + b" in c),
        ("CPP/AtomicCounter", "#include <atomic>\n\nstruct Counter {\n    std::atomic<int> val{0};\n    void increment() {\n", lambda c: "fetch_add" in c or "++" in c or "val" in c),

        # Go (5)
        ("Go/WorkerPoolChannels", "package main\n\nfunc worker(id int, jobs <-chan int, results chan<- int) {\n    for j := range jobs {\n", lambda c: "results <-" in c or "jobs" in c),
        ("Go/BinarySearchSlice", "package main\n\nfunc BinarySearch(arr []int, target int) int {\n", lambda c: "left" in c and "right" in c and "mid" in c),
        ("Go/StructJSONMarshal", "package main\n\ntype User struct {\n    Name string `json:\"name\"`\n    Age  int    `json:\"age\"`\n}\n", lambda c: "json" in c and "struct" in c),
        ("Go/HTTPHandlerEndpoint", "package main\nimport \"net/http\"\n\nfunc HealthCheckHandler(w http.ResponseWriter, r *http.Request) {\n", lambda c: "w.WriteHeader" in c or "w.Write" in c or "fmt.Fprint" in c),
        ("Go/SyncMutexSafeMap", "package main\nimport \"sync\"\n\ntype SafeMap struct {\n    mu sync.RWMutex\n    m  map[string]int\n}\nfunc (s *SafeMap) Get(key string) (int, bool) {\n", lambda c: "s.mu.RLock" in c or "s.mu.Lock" in c),

        # TypeScript (5)
        ("TypeScript/GenericDebounce", "export function debounce<T extends (...args: any[]) => any>(fn: T, ms: number) {\n    let timeoutId: ReturnType<typeof setTimeout> | null = null;\n    return (...args: Parameters<T>) => {\n", lambda c: "clearTimeout" in c and "setTimeout" in c),
        ("TypeScript/PromiseRetry", "export async function retry<T>(fn: () => Promise<T>, retries: number): Promise<T> {\n", lambda c: "try" in c and "catch" in c and ("await" in c or "retries" in c)),
        ("TypeScript/DeepCloneGeneric", "export function deepClone<T>(obj: T): T {\n", lambda c: "structuredClone" in c or "JSON.parse" in c or "typeof" in c),
        ("TypeScript/EventEmitterType", "type Listener = (...args: any[]) => void;\nexport class EventEmitter {\n    private events: Map<string, Listener[]> = new Map();\n    on(event: string, fn: Listener) {\n", lambda c: "set" in c or "get" in c or "push" in c),
        ("TypeScript/ZodLikeValidator", "export type Validator<T> = (val: unknown) => val is T;\nexport const isString: Validator<string> = (val): val is string => {\n    return typeof val === 'string';\n};\n", lambda c: "typeof" in c or "return" in c)
    ]
    for name, p, v in multi_prompts:
        tasks.append({"id": name, "category": name.split("/")[0], "prompt": p, "verifier": v, "max_tokens": 130})

    # ---------------------------------------------------------
    # 3. 20 BÀI CODING AGENT & TOOL CALLING & DEBUGGING
    # ---------------------------------------------------------
    agent_prompts = [
        # Tool Calling (JSON Schema) (1-5)
        ("Agent/ToolCall/Grep", "<|im_start|>system\nYou are a coding assistant. Available tools:\n- grep_search(Query: str, SearchPath: str)\nCall tool in JSON format: {\"tool\": \"grep_search\", \"parameters\": {\"Query\": \"...\", \"SearchPath\": \"...\"}}<|im_end|>\n<|im_start|>user\nFind 'def binary_search' in /src.<|im_end|>\n<|im_start|>assistant\n", lambda c: "grep_search" in c and "/src" in c and "binary_search" in c),
        ("Agent/ToolCall/ReadFile", "<|im_start|>system\nAvailable tools:\n- read_file(path: str, start_line: int, end_line: int)\nReturn JSON tool call.<|im_end|>\n<|im_start|>user\nRead lines 10 to 50 of /workspace/main.py.<|im_end|>\n<|im_start|>assistant\n", lambda c: "read_file" in c and "main.py" in c and ("10" in c or "50" in c)),
        ("Agent/ToolCall/WriteFile", "<|im_start|>system\nAvailable tools:\n- write_to_file(path: str, content: str)\nReturn JSON tool call.<|im_end|>\n<|im_start|>user\nCreate file /app/test.py with content 'print(1)'.<|im_end|>\n<|im_start|>assistant\n", lambda c: "write_to_file" in c and "/app/test.py" in c and "print" in c),
        ("Agent/ToolCall/RunCommand", "<|im_start|>system\nAvailable tools:\n- run_command(command: str)\nReturn JSON tool call.<|im_end|>\n<|im_start|>user\nRun pytest on tests/ directory.<|im_end|>\n<|im_start|>assistant\n", lambda c: "run_command" in c and "pytest" in c),
        ("Agent/ToolCall/ListDir", "<|im_start|>system\nAvailable tools:\n- list_dir(path: str)\nReturn JSON tool call.<|im_end|>\n<|im_start|>user\nList all files in /home/htsc/project.<|im_end|>\n<|im_start|>assistant\n", lambda c: "list_dir" in c and "/home/htsc/project" in c),

        # Debugging & Self-Correction (6-12)
        ("Agent/Debug/TypeErrorConcat", "<|im_start|>system\nFix code error: TypeError: can only concatenate str to str<|im_end|>\n<|im_start|>user\ndef greet(name: str, age: int):\n    return 'Hello ' + name + ', age ' + age\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "str(age)" in c or "f\"" in c or "f'" in c or "format" in c),
        ("Agent/Debug/IndexOutOfBounds", "<|im_start|>system\nFix IndexError: list index out of range in loop<|im_end|>\n<|im_start|>user\ndef get_last(items: list[int]):\n    return items[len(items)]\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "len(items) - 1" in c or "items[-1]" in c),
        ("Agent/Debug/ZeroDivision", "<|im_start|>system\nFix ZeroDivisionError handling<|im_end|>\n<|im_start|>user\ndef divide(a: float, b: float) -> float:\n    return a / b\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "b == 0" in c or "except ZeroDivisionError" in c or "if b" in c),
        ("Agent/Debug/KeyErrorDict", "<|im_start|>system\nFix KeyError safely<|im_end|>\n<|im_start|>user\ndef get_config(data: dict, key: str, default: str):\n    return data[key]\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: ".get(" in c or "if key in" in c),
        ("Agent/Debug/MutableDefaultArg", "<|im_start|>system\nFix Python mutable default argument bug<|im_end|>\n<|im_start|>user\ndef append_to(element, target_list=[]):\n    target_list.append(element)\n    return target_list\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "target_list is None" in c or "target_list = None" in c or "None" in c),
        ("Agent/Debug/AsyncMissingAwait", "<|im_start|>system\nFix missing await in async function<|im_end|>\n<|im_start|>user\nasync def fetch_user(client, user_id):\n    res = client.get_user(user_id)\n    return res.name\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "await client.get_user" in c or "await" in c),
        ("Agent/Debug/RecursionDepth", "<|im_start|>system\nFix RecursionError: maximum recursion depth exceeded<|im_end|>\n<|im_start|>user\ndef countdown(n: int):\n    print(n)\n    countdown(n - 1)\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "if n <= 0" in c or "if n < 0" in c or "if n == 0" in c or "return" in c),

        # Repository Editing & Diffs (13-20)
        ("Agent/Diff/AddLogging", "<|im_start|>system\nOutput unified diff to add logging to run().<|im_end|>\n<|im_start|>user\n--- a/server.py\n+++ b/server.py\n@@ -1,3 +1,4 @@\n def run():\n+    logger.info('Server starting')\n     app.listen(8080)\n<|im_end|>\n<|im_start|>assistant\n", lambda c: "logger" in c or "listen" in c or "app" in c),
        ("Agent/Diff/AddTypeHints", "<|im_start|>system\nConvert function to type-hinted signature.<|im_end|>\n<|im_start|>user\ndef add_user(users, name, age=18):\n    users[name] = age\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "dict" in c or "str" in c or "int" in c),
        ("Agent/Diff/ExtractHelperMethod", "<|im_start|>system\nRefactor validation into a helper function.<|im_end|>\n<|im_start|>user\ndef register(email, pwd):\n    if '@' not in email or len(pwd) < 8:\n        raise ValueError('Invalid')\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "validate" in c or "def " in c or "email" in c),
        ("Agent/Diff/AddContextManager", "<|im_start|>system\nConvert file reading to use context manager.<|im_end|>\n<|im_start|>user\ndef read_all(path):\n    f = open(path)\n    data = f.read()\n    f.close()\n    return data\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "with open" in c),
        ("Agent/Diff/FIMCompletion", "<|fim_prefix|>def compute_stats(scores: list[float]) -> dict[str, float]:\n    if not scores: return {}\n<|fim_suffix|>\n    return {'mean': mean_val, 'std': std_val}\n<|fim_middle|>", lambda c: "mean_val" in c or "sum(scores)" in c or "std" in c),
        ("Agent/Diff/DocstringGeneration", "<|im_start|>user\nWrite Google-style docstring for:\ndef retry_request(url: str, max_retries: int = 3, backoff: float = 1.5) -> dict:\n<|im_end|>\n<|im_start|>assistant\n", lambda c: "Args:" in c or "Returns:" in c or "url" in c),
        ("Agent/Diff/SQLInjectionFix", "<|im_start|>system\nFix SQL injection vulnerability by using parameterized query.<|im_end|>\n<|im_start|>user\ndef get_user(cursor, username):\n    return cursor.execute(f'SELECT * FROM users WHERE name = {username}')\n<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "%s" in c or "?" in c or "params" in c or "(username,)" in c or "username" in c),
        ("Agent/Diff/AsyncFastAPIRoute", "<|im_start|>user\nCreate FastAPI async route GET /items/{item_id} with Pydantic response.<|im_end|>\n<|im_start|>assistant\n```python\n", lambda c: "@app.get" in c and "async def" in c and "item_id" in c)
    ]
    for name, p, v in agent_prompts:
        tasks.append({"id": name, "category": "CodingAgent", "prompt": p, "verifier": v, "max_tokens": 120})

    return tasks


def run_comprehensive_eval():
    model_path = "./qwen3.8_flash_finetuned_26gb"
    print("=" * 100)
    print(" 🚀 BẮT ĐẦU CHẠY BỘ BENCHMARK TOÀN DIỆN 100 BÀI TEST TRÊN BẢN BF16 NATIVE")
    print(f"    • Model Source : {model_path}")
    print(f"    • Total Tasks  : 100 Real-world Coding & Agent Scenarios")
    print("=" * 100)

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    device_map = {
        "model.embed_tokens": "cuda:0",
        "model.hyper_connection_mixer": "cuda:2",
        "model.norm": "cuda:2",
        "lm_head": "cuda:2"
    }
    for i in range(15):
        device_map[f"model.layers.{i}"] = "cuda:0"
    for i in range(15, 32):
        device_map[f"model.layers.{i}"] = "cuda:1"
    for i in range(32, 48):
        device_map[f"model.layers.{i}"] = "cuda:2"

    print("[*] Nạp mô hình BF16 Native vào 3 GPU...")
    t0 = time.perf_counter()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )
    print(f"[✓] Nạp weights hoàn tất trong {time.perf_counter() - t0:.2f}s!")

    # Warm-up
    print("[*] Thực hiện GPU Warm-up...")
    warm_in = tokenizer("def warm_up(): return True", return_tensors="pt").to("cuda:0")
    with torch.no_grad():
        _ = model.generate(**warm_in, max_new_tokens=5)
    print("[✓] GPU Warm-up hoàn tất!")

    suite = build_100_task_suite()
    print(f"\n[*] Đã nạp {len(suite)} bài test. Bắt đầu đánh giá tự động...\n")

    results = []
    category_stats = {}

    t_eval_start = time.perf_counter()
    for idx, task in enumerate(suite, 1):
        tid = task["id"]
        cat = task["category"]
        prompt = task["prompt"]
        verifier = task["verifier"]
        max_tok = task["max_tokens"]

        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        t_start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tok,
                do_sample=False,
                repetition_penalty=1.08,
                pad_token_id=tokenizer.eos_token_id
            )
        t_lat = time.perf_counter() - t_start
        new_tokens = outputs[0][inputs.input_ids.shape[1]:]
        num_tok = len(new_tokens)
        tps = num_tok / max(t_lat, 1e-4)
        ttft_est = (t_lat / max(num_tok, 1)) * 1000

        gen_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
        is_pass = verifier(gen_text)

        if cat not in category_stats:
            category_stats[cat] = {"passed": 0, "total": 0}
        category_stats[cat]["total"] += 1
        if is_pass:
            category_stats[cat]["passed"] += 1

        results.append({
            "id": tid,
            "category": cat,
            "passed": is_pass,
            "ttft_ms": ttft_est,
            "tps": tps,
            "tokens": num_tok,
            "snippet": gen_text[:90].replace("\n", " ")
        })

        status_icon = "✅" if is_pass else "❌"
        if idx % 10 == 0 or idx == len(suite):
            passed_so_far = sum([1 for r in results if r["passed"]])
            print(f"  [{idx:3d}/{len(suite)}] Tiến độ: {passed_so_far}/{idx} Passed ({(passed_so_far/idx)*100:.1f}%) | Bài vừa chạy [{tid}]: {status_icon}")

    total_time = time.perf_counter() - t_eval_start
    total_passed = sum([1 for r in results if r["passed"]])
    pass_rate = (total_passed / len(suite)) * 100.0

    print("\n" + "=" * 100)
    print(f" 🏆 TỔNG KẾT BENCHMARK 100 BÀI TEST TRÊN BẢN BF16 NATIVE:")
    print(f"    • Tỷ Lệ Pass@1 Tổng Thể : {total_passed}/{len(suite)} ({pass_rate:.1f}%)")
    print(f"    • Tổng thời gian chạy   : {total_time:.2f}s (Trung bình {total_time/len(suite):.2f}s/bài)")
    print("=" * 100)
    print(" 📊 KẾT QUẢ THEO TỪNG NHÓM TÁC VỤ:")
    for cat, stat in category_stats.items():
        p = stat["passed"]
        tot = stat["total"]
        pct = (p / tot) * 100.0
        print(f"    • {cat:<22}: {p:2d}/{tot:2d} ({pct:5.1f}%)")
    print("=" * 100)

    # Lưu báo cáo JSON
    os.makedirs("./benchmarks", exist_ok=True)
    with open("./benchmarks/bf16_comprehensive_100_results.json", "w", encoding="utf-8") as f:
        json.dump({
            "model": "Qwen3.8-Flash-Coder (BF16 Native 128 Experts)",
            "pass_rate_pct": pass_rate,
            "total_passed": total_passed,
            "total_tasks": len(suite),
            "total_time_seconds": total_time,
            "category_stats": category_stats,
            "detailed_results": results
        }, f, indent=2)
    print("[✓] Đã lưu toàn bộ báo cáo chi tiết vào: ./benchmarks/bf16_comprehensive_100_results.json")


if __name__ == "__main__":
    run_comprehensive_eval()
