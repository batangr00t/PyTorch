import heapq
import time
import threading
from queue import PriorityQueue
from collections import deque
import random

def benchmark_heapq(n=10000):
    """heapq 성능 측정"""
    heap = []
    
    # 삽입 시간 측정
    start = time.time()
    for i in range(n):
        heapq.heappush(heap, random.randint(1, 1000))
    insert_time = time.time() - start
    
    # 삭제 시간 측정
    start = time.time()
    while heap:
        heapq.heappop(heap)
    pop_time = time.time() - start
    
    return insert_time, pop_time

def benchmark_priority_queue(n=10000):
    """PriorityQueue 성능 측정"""
    pq = PriorityQueue()
    
    # 삽입 시간 측정
    start = time.time()
    for i in range(n):
        pq.put(random.randint(1, 1000))
    insert_time = time.time() - start
    
    # 삭제 시간 측정
    start = time.time()
    while not pq.empty():
        pq.get()
    pop_time = time.time() - start
    
    return insert_time, pop_time

def thread_safety_test():
    """스레드 안전성 테스트"""
    # heapq (스레드 안전하지 않음)
    heap = []
    pq = PriorityQueue()
    
    def heapq_worker():
        for i in range(1000):
            heapq.heappush(heap, i)
            if heap:
                heapq.heappop(heap)
    
    def pq_worker():
        for i in range(1000):
            pq.put(i)
            if not pq.empty():
                try:
                    pq.get_nowait()
                except:
                    pass
    
    # heapq 멀티스레드 테스트
    threads = []
    for _ in range(3):
        t = threading.Thread(target=heapq_worker)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print(f"heapq 멀티스레드 후 크기: {len(heap)}")
    
    # PriorityQueue 멀티스레드 테스트
    threads = []
    for _ in range(3):
        t = threading.Thread(target=pq_worker)
        threads.append(t)
        t.start()
    
    for t in threads:
        t.join()
    
    print(f"PriorityQueue 멀티스레드 후 크기: {pq.qsize()}")

def memory_usage_comparison():
    """메모리 사용량 비교 (간접적)"""
    import sys
    
    # heapq
    heap = []
    for i in range(10000):
        heapq.heappush(heap, i)
    heapq_size = sys.getsizeof(heap)
    
    # PriorityQueue
    pq = PriorityQueue()
    for i in range(10000):
        pq.put(i)
    pq_size = sys.getsizeof(pq) + sys.getsizeof(pq.queue)
    
    return heapq_size, pq_size

def feature_comparison():
    """기능 비교 데모"""
    print("=== 기능 비교 ===")
    
    # heapq 기능들
    heap = [3, 1, 4, 1, 5, 9, 2]
    print(f"원본 리스트: {heap}")
    
    heapq.heapify(heap)
    print(f"heapify 후: {heap}")
    
    print(f"3개 최소값: {heapq.nsmallest(3, [3, 1, 4, 1, 5, 9, 2])}")
    print(f"3개 최대값: {heapq.nlargest(3, [3, 1, 4, 1, 5, 9, 2])}")
    
    # 최소값 확인 (제거 안함)
    print(f"최소값 확인: {heap[0]}")
    
    # PriorityQueue는 이런 기능들이 없음
    pq = PriorityQueue()
    pq.put(3)
    pq.put(1)
    pq.put(4)
    
    print(f"PriorityQueue 크기: {pq.qsize()}")
    # print(f"PriorityQueue 최소값 확인: ???")  # 불가능

if __name__ == "__main__":
    print("PriorityQueue vs heapq 비교\n")
    
    # 성능 비교
    print("=== 성능 비교 (10,000개 요소) ===")
    
    # 여러 번 측정하여 평균 계산
    heapq_insert_times = []
    heapq_pop_times = []
    pq_insert_times = []
    pq_pop_times = []
    
    for _ in range(5):
        hi, hp = benchmark_heapq(10000)
        heapq_insert_times.append(hi)
        heapq_pop_times.append(hp)
        
        pi, pp = benchmark_priority_queue(10000)
        pq_insert_times.append(pi)
        pq_pop_times.append(pp)
    
    avg_heapq_insert = sum(heapq_insert_times) / len(heapq_insert_times)
    avg_heapq_pop = sum(heapq_pop_times) / len(heapq_pop_times)
    avg_pq_insert = sum(pq_insert_times) / len(pq_insert_times)
    avg_pq_pop = sum(pq_pop_times) / len(pq_pop_times)
    
    print(f"heapq 삽입 시간: {avg_heapq_insert:.4f}초")
    print(f"heapq 삭제 시간: {avg_heapq_pop:.4f}초")
    print(f"PriorityQueue 삽입 시간: {avg_pq_insert:.4f}초")
    print(f"PriorityQueue 삭제 시간: {avg_pq_pop:.4f}초")
    print(f"heapq가 삽입에서 {avg_pq_insert/avg_heapq_insert:.1f}배 빠름")
    print(f"heapq가 삭제에서 {avg_pq_pop/avg_heapq_pop:.1f}배 빠름")
    
    print("\n=== 메모리 사용량 비교 ===")
    heapq_mem, pq_mem = memory_usage_comparison()
    print(f"heapq 메모리: {heapq_mem:,} bytes")
    print(f"PriorityQueue 메모리: {pq_mem:,} bytes")
    print(f"heapq가 {pq_mem/heapq_mem:.1f}배 적은 메모리 사용")
    
    print("\n=== 스레드 안전성 테스트 ===")
    thread_safety_test()
    
    print()
    feature_comparison()