export async function uploadAndGetScore(file) {
  const formData = new FormData();
  formData.append("image", file);

  const controller = new AbortController();
  const timeoutMs = 5000; // 5 seconds

  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const res = await fetch("http://localhost:5000/score", {
      method: "POST",
      body: formData,
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!res.ok) {
      throw new Error(`Server responded with ${res.status}`);
    }

    return await res.json();
  } catch (err) {
    // If the request was aborted due to timeout, or a network error occurred,
    // return mock data so the UI can continue to work offline or when the
    // backend is unreachable.
    if (err.name === 'AbortError' || err.name === 'TypeError' || /Failed to fetch/.test(err.message)) {
      // Mock response -- adjust fields to match your real API shape
      return {
        success: false,
        note: 'returned mock data due to timeout',
        score: 72,
        breakdown: {
          composition: 75,
          lighting: 70,
          clarity: 68,
          story: 78,
        },
      };
    }

    // For other errors, rethrow so caller can handle them
    throw err;
  }
}
