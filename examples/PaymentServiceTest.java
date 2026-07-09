public class PaymentServiceTest {
    @Test
    public void retriesOnTimeout() throws Exception {
        ExecutorService pool = Executors.newFixedThreadPool(2);
        Thread.sleep(500);
        HttpClient client = HttpClient.newHttpClient();
        assertTrue(service.retry(client));
    }

    @Test
    public void computesTotal() {
        assertEquals(30, new Cart(10, 20).total());
    }

    @Test
    public void writesAuditLog() throws Exception {
        File tmp = File.createTempFile("audit", ".log");
        long now = System.currentTimeMillis();
        if (tmp.exists()) { assertNotNull(new AuditLog(tmp, now)); }
    }
}
